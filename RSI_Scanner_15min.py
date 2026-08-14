import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

SCRIPT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_FOLDER / "My-Stocks.csv"
REPORT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = SCRIPT_FOLDER / f"RSI_Scanner_{REPORT_TIMESTAMP}.csv"
LATEST_OUTPUT_FILE = SCRIPT_FOLDER / "latest_results.csv"
LATEST_JSON_FILE = SCRIPT_FOLDER / "latest_results.json"

# Persistent RSI history. Every scan appends one row per stock.
HISTORY_FILE = SCRIPT_FOLDER / "RSI_History.csv"

RSI_PERIOD = 14
WEEKLY_LOOKBACK_DAYS = 730
HOURLY_LOOKBACK_DAYS = 30
UPSTOX_API = "https://api.upstox.com"

# Read the Upstox token from an environment variable.
# GitHub Actions supplies this from the repository secret:
# UPSTOX_ACCESS_TOKEN
import os

ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()

if not ACCESS_TOKEN:
    raise SystemExit(
        "ERROR: UPSTOX_ACCESS_TOKEN is not configured."
    )

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}


def calculate_rsi(close, period=14):
    close = pd.to_numeric(close, errors="coerce").reset_index(drop=True)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi = pd.Series(index=close.index, dtype=float)

    if len(close) <= period:
        return rsi

    avg_gain = gain.iloc[1:period + 1].mean()
    avg_loss = loss.iloc[1:period + 1].mean()

    if avg_loss == 0:
        rsi.iloc[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rsi.iloc[period] = 100 - (100 / (1 + avg_gain / avg_loss))

    for i in range(period + 1, len(close)):
        avg_gain = ((avg_gain * (period - 1)) + gain.iloc[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + loss.iloc[i]) / period

        if avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rsi.iloc[i] = 100 - (100 / (1 + avg_gain / avg_loss))

    return rsi


def load_stocks():
    """
    Load stocks from My-Stocks.csv.

    Required column:
        Symbol

    Optional column:
        Tag

    Example:
        Symbol,Tag
        NSE:AARTIIND,52 Week High
        NSE:ABB,Volume shockers
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"My-Stocks.csv not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "Symbol" not in df.columns:
        raise ValueError(
            "My-Stocks.csv must contain a column named 'Symbol'."
        )

    # Tag is optional so an older My-Stocks.csv will still work.
    if "Tag" not in df.columns:
        df["Tag"] = ""

    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("\\", "", regex=False)
    )

    df["Tag"] = (
        df["Tag"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[df["Symbol"] != ""].copy()

    # Keep the first row if a symbol appears more than once.
    df = df.drop_duplicates(
        subset=["Symbol"],
        keep="first"
    )

    return df


def find_instrument_key(symbol):
    trading_symbol = symbol.replace("NSE:", "").strip().upper()

    response = requests.get(
        f"{UPSTOX_API}/v2/instruments/search",
        headers=HEADERS,
        params={
            "query": trading_symbol,
            "exchanges": "NSE",
            "segments": "EQ",
            "page_number": 1,
            "records": 30,
        },
        timeout=20,
    )

    if response.status_code != 200:
        print(f"  Instrument search failed: {response.status_code}")
        return None

    instruments = response.json().get("data", [])

    for item in instruments:
        if (
            str(item.get("trading_symbol", "")).upper() == trading_symbol
            and item.get("segment") == "NSE_EQ"
        ):
            return item.get("instrument_key")

    for item in instruments:
        if item.get("segment") == "NSE_EQ":
            return item.get("instrument_key")

    return None


def get_candles(instrument_key, unit, lookback_days):
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (
        datetime.now() - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    encoded_key = quote(instrument_key, safe="")

    response = requests.get(
        f"{UPSTOX_API}/v3/historical-candle/"
        f"{encoded_key}/{unit}/1/{to_date}/{from_date}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(f"  {unit} candle request failed: {response.status_code}")
        return []

    return response.json().get("data", {}).get("candles", [])


def get_current_ltp(instrument_key):
    encoded_key = quote(instrument_key, safe="")

    response = requests.get(
        f"{UPSTOX_API}/v3/market-quote/ltp"
        f"?instrument_key={encoded_key}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(f"  LTP request failed: {response.status_code}")
        return None

    quote_data = response.json().get("data", {})

    if not quote_data:
        return None

    first_quote = next(iter(quote_data.values()))
    last_price = first_quote.get("last_price")

    return float(last_price) if last_price is not None else None


def candle_dataframe(candles):
    rows = [
        {"timestamp": c[0], "close": c[4]}
        for c in candles
        if len(c) >= 5
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def get_current_weekly_rsi(instrument_key, ltp):
    df = candle_dataframe(
        get_candles(instrument_key, "weeks", WEEKLY_LOOKBACK_DAYS)
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    week_start = (now - pd.Timedelta(days=now.weekday())).normalize()

    local_time = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    completed = df[local_time < week_start].copy()

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([{
        "timestamp": week_start,
        "close": ltp,
    }])

    calc = pd.concat([completed[["timestamp", "close"]], current], ignore_index=True)
    calc["RSI"] = calculate_rsi(calc["close"], RSI_PERIOD)
    valid = calc.dropna(subset=["RSI"])

    if len(valid) < 2:
        return None

    return {
        "current": float(valid.iloc[-1]["RSI"]),
        "previous": float(valid.iloc[-2]["RSI"]),
    }


def get_current_hourly_rsi(instrument_key, ltp):
    df = candle_dataframe(
        get_candles(instrument_key, "hours", HOURLY_LOOKBACK_DAYS)
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    current_hour = now.floor("h")

    local_time = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    completed = df[local_time < current_hour].copy()

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([{
        "timestamp": current_hour,
        "close": ltp,
    }])

    calc = pd.concat([completed[["timestamp", "close"]], current], ignore_index=True)
    calc["RSI"] = calculate_rsi(calc["close"], RSI_PERIOD)
    valid = calc.dropna(subset=["RSI"])

    if len(valid) < 2:
        return None

    return {
        "current": float(valid.iloc[-1]["RSI"]),
        "previous": float(valid.iloc[-2]["RSI"]),
        "hour": current_hour.strftime("%Y-%m-%d %H:%M"),
    }


def classify(weekly, hourly):
    weekly_rsi = weekly["current"]
    hourly_rsi = hourly["current"]
    previous_hourly = hourly["previous"]
    rising = hourly_rsi > previous_hourly

    if weekly_rsi <= 50:
        return "IGNORE", "❌ IGNORE", "Weekly RSI <= 50"

    if hourly_rsi > 50:
        return "WAIT", "⏳ WAIT", "Hourly RSI > 50"

    if 30 <= hourly_rsi <= 50:
        if rising:
            return "WATCH", "👀 WATCH", "Hourly RSI 30-50 and rising"
        return "WATCH", "👀 WATCH", "Hourly RSI 30-50"

    if hourly_rsi < 30:
        if rising:
            return (
                "SETUP",
                "🔥 SETUP",
                "Weekly RSI > 50 + Hourly RSI < 30 + Hourly RSI rising",
            )
        return "WATCH", "👀 WATCH", "Hourly RSI < 30 but still falling"

    return "WAIT", "⏳ WAIT", "No setup"


def save_rsi_history(results):
    """Append the current RSI snapshot to RSI_History.csv."""

    if not results:
        return

    history_columns = [
        "Scan Time",
        "Symbol",
        "Current LTP",
        "Current Week RSI",
        "Previous Week RSI",
        "Weekly RSI Change",
        "Current Hour",
        "Current Hourly RSI",
        "Previous Hourly RSI",
        "Hourly RSI Change",
        "Hourly RSI Rising",
        "History Transition",
        "Category",
        "Signal",
    ]

    now_ist = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    history_rows = []

    for result in results:
        history_rows.append({
            "Scan Time": now_ist,
            "Symbol": result["Symbol"],
            "Current LTP": result["Current LTP"],
            "Current Week RSI": result["Current Week RSI"],
            "Previous Week RSI": result["Previous Week RSI"],
            "Weekly RSI Change": result["Weekly RSI Change"],
            "Current Hour": result["Current Hour"],
            "Current Hourly RSI": result["Current Hourly RSI"],
            "Previous Hourly RSI": result["Previous Hourly RSI"],
            "Hourly RSI Change": result["Hourly RSI Change"],
            "Hourly RSI Rising": result["Hourly RSI Rising"],
            "Category": result["Category"],
            "Signal": result["Signal"],
        })

    new_history = pd.DataFrame(
        history_rows,
        columns=history_columns,
    )

    if HISTORY_FILE.exists():
        new_history.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        new_history.to_csv(
            HISTORY_FILE,
            mode="w",
            header=True,
            index=False,
            encoding="utf-8-sig",
        )


def get_previous_scan_for_symbol(symbol):
    """
    Return the most recent saved scan for a symbol, if available.
    Used to detect RSI direction across separate scanner runs.
    """

    if not HISTORY_FILE.exists():
        return None

    try:
        history = pd.read_csv(
            HISTORY_FILE,
            usecols=[
                "Scan Time",
                "Symbol",
                "Current Hourly RSI",
                "Current Week RSI",
                "Category",
                "Signal",
            ],
        )
    except Exception:
        return None

    if history.empty:
        return None

    rows = history[
        history["Symbol"].astype(str).str.upper() == symbol.upper()
    ]

    if rows.empty:
        return None

    return rows.iloc[-1].to_dict()


def detect_history_transition(symbol, current_hourly_rsi, current_weekly_rsi):
    """Detect a real oversold reversal from 15-minute scan history."""
    previous = get_previous_scan_for_symbol(symbol)

    if previous is None:
        return "FIRST SCAN", False, None, None

    previous_rsi = pd.to_numeric(
        previous.get("Current Hourly RSI"), errors="coerce"
    )
    if pd.isna(previous_rsi):
        return "NO PREVIOUS RSI", False, None, None

    # Load the last two historical observations for this symbol.
    try:
        history = pd.read_csv(HISTORY_FILE)
        rows = history[history["Symbol"].astype(str).str.upper() == symbol.upper()].copy()
        if "Scan Time" in rows.columns:
            rows["Scan Time"] = pd.to_datetime(rows["Scan Time"], errors="coerce")
            rows = rows.sort_values("Scan Time")
        rows = rows.tail(2)
    except Exception:
        rows = pd.DataFrame()

    scan_before_rsi = None
    if len(rows) >= 2:
        scan_before_rsi = pd.to_numeric(
            rows.iloc[-2].get("Current Hourly RSI"), errors="coerce"
        )
        if pd.isna(scan_before_rsi):
            scan_before_rsi = None

    # Weekly filter first.
    if current_weekly_rsi <= 50:
        return "WEEKLY RSI <= 50", False, previous_rsi, scan_before_rsi

    change = current_hourly_rsi - previous_rsi

    # Entered oversold.
    if current_hourly_rsi < 30 and previous_rsi >= 30:
        return "ENTERED OVERSOLD", False, previous_rsi, scan_before_rsi

    # Oversold reversal: previous scan was below 30 and current RSI rises.
    if current_hourly_rsi < 30 and previous_rsi < 30 and change > 0:
        # Fresh reversal if the scan before the previous one was flat/falling,
        # or if there is no older scan available.
        if scan_before_rsi is None or previous_rsi <= scan_before_rsi:
            return "🔥 OVERSOLD REVERSAL", True, previous_rsi, scan_before_rsi
        return "OVERSOLD + RISING", False, previous_rsi, scan_before_rsi

    if current_hourly_rsi < 30 and change < 0:
        return "OVERSOLD + FALLING", False, previous_rsi, scan_before_rsi

    if current_hourly_rsi < 30:
        return "OVERSOLD + FLAT", False, previous_rsi, scan_before_rsi

    if previous_rsi < 30 and current_hourly_rsi >= 30:
        return "EXITED OVERSOLD", False, previous_rsi, scan_before_rsi

    if 30 <= current_hourly_rsi <= 50:
        if change > 0:
            return "30-50 + RISING", False, previous_rsi, scan_before_rsi
        if change < 0:
            return "30-50 + FALLING", False, previous_rsi, scan_before_rsi
        return "30-50 + FLAT", False, previous_rsi, scan_before_rsi

    if change > 0:
        transition = "ABOVE 50 + RISING"
    elif change < 0:
        transition = "ABOVE 50 + FALLING"
    else:
        transition = "ABOVE 50 + FLAT"

    return transition, False, previous_rsi, scan_before_rsi

def process_stock(symbol, tag=""):
    print(f"\nChecking {symbol} ...")
    if tag:
        print(f"  Tag: {tag}")

    instrument_key = find_instrument_key(symbol)
    if not instrument_key:
        return None

    ltp = get_current_ltp(instrument_key)
    if ltp is None:
        return None

    weekly = get_current_weekly_rsi(instrument_key, ltp)
    hourly = get_current_hourly_rsi(instrument_key, ltp)

    if weekly is None or hourly is None:
        return None

    category, signal, reason = classify(weekly, hourly)

    hourly_change = hourly["current"] - hourly["previous"]

    history_transition, history_setup, previous_scan_rsi, scan_before_rsi = detect_history_transition(
        symbol,
        hourly["current"],
        weekly["current"],
    )

    print(f"  LTP: ₹{ltp:.2f}")
    print(f"  Weekly RSI: {weekly['current']:.2f}")
    print(f"  Hourly RSI: {hourly['current']:.2f}")
    print(f"  Hourly Change: {hourly_change:+.2f}")
    print(f"  History Transition: {history_transition}")
    print(f"  Signal: {signal}")

    return {
        "Symbol": symbol,
        "Tag": tag,
        "Instrument Key": instrument_key,
        "Current LTP": round(ltp, 2),
        "Current Week RSI": round(weekly["current"], 2),
        "Previous Week RSI": round(weekly["previous"], 2),
        "Weekly RSI Change": round(weekly["current"] - weekly["previous"], 2),
        "Current Hour": hourly["hour"],
        "Current Hourly RSI": round(hourly["current"], 2),
        "Previous Hourly RSI": round(hourly["previous"], 2),
        "Hourly RSI Change": round(hourly_change, 2),
        "Hourly RSI Rising": "YES" if hourly_change > 0 else "NO",
        "15-Min Previous RSI": (
            round(previous_scan_rsi, 2)
            if previous_scan_rsi is not None else ""
        ),
        "15-Min RSI Change": (
            round(hourly["current"] - previous_scan_rsi, 2)
            if previous_scan_rsi is not None else ""
        ),
        "History Transition": history_transition,
        "Category": category,
        "Signal": signal,
        "Reason": reason,
    }


def main():
    print("=" * 90)
    print("RSI SCANNER")
    print("Weekly RSI > 50 + Hourly RSI Pullback + Hourly RSI Reversal")
    print("=" * 90)

    stocks = load_stocks()
    print(f"Stocks found: {len(stocks)}")

    results = []

    # Read both Symbol and Tag from the Google-Sheets-generated CSV.
    stock_rows = stocks.to_dict("records")

    for number, row in enumerate(stock_rows, start=1):
        symbol = row["Symbol"]
        tag = row.get("Tag", "")

        print(f"\n[{number}/{len(stock_rows)}] {symbol}")

        if tag:
            print(f"  Tag: {tag}")

        try:
            result = process_stock(symbol, tag)
            if result is not None:
                results.append(result)
        except Exception as error:
            print(f"  ERROR: {error}")

        time.sleep(0.3)

    if not results:
        print("\nNo results generated.")
        return

    output = pd.DataFrame(results)

    priority = {"SETUP": 1, "WATCH": 2, "WAIT": 3, "IGNORE": 4}
    output["_priority"] = output["Category"].map(priority).fillna(99)

    output = (
        output.sort_values(
            ["_priority", "Current Hourly RSI"],
            ascending=[True, True],
        )
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    # Save timestamped archive report.
    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Save latest scan using stable filenames for GitHub Pages / Actions.
    output.to_csv(
        LATEST_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    output.to_json(
        LATEST_JSON_FILE,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # Persist this scan for future comparisons.
    save_rsi_history(results)

    setup_count = int((output["Category"] == "SETUP").sum())
    watch_count = int((output["Category"] == "WATCH").sum())
    wait_count = int((output["Category"] == "WAIT").sum())
    ignore_count = int((output["Category"] == "IGNORE").sum())

    print("\n" + "=" * 90)
    print("FINAL RSI SCANNER RESULT")
    print("=" * 90)

    columns = [
        "Symbol",
        "Current Week RSI",
        "Current Hourly RSI",
        "Hourly RSI Change",
        "Hourly RSI Rising",
        "Category",
        "Signal",
        "Reason",
    ]

    print(output[columns].to_string(index=False))

    print("\n" + "=" * 90)
    print(f"🔥 SETUP : {setup_count}")
    print(f"👀 WATCH : {watch_count}")
    print(f"⏳ WAIT  : {wait_count}")
    print(f"❌ IGNORE: {ignore_count}")
    print(f"\nReport saved: {OUTPUT_FILE}")
    print(f"Latest report: {LATEST_OUTPUT_FILE}")
    print(f"Latest JSON: {LATEST_JSON_FILE}")
    print(f"History saved/appended: {HISTORY_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
