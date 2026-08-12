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

# Stable files consumed by GitHub Pages / index.html.
LATEST_CSV_FILE = SCRIPT_FOLDER / "latest_results.csv"
LATEST_JSON_FILE = SCRIPT_FOLDER / "latest_results.json"

# Persistent RSI history. Every scan appends one row per stock.
HISTORY_FILE = SCRIPT_FOLDER / "RSI_History.csv"

RSI_PERIOD = 14
WEEKLY_LOOKBACK_DAYS = 730
HOURLY_LOOKBACK_DAYS = 30
FIFTEEN_MIN_LOOKBACK_DAYS = 10
UPSTOX_API = "https://api.upstox.com"

# Upstox token is supplied by GitHub Actions as a repository secret.
import os

ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()

if not ACCESS_TOKEN:
    raise SystemExit("ERROR: UPSTOX_ACCESS_TOKEN is not configured.")

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
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"My-Stocks.csv not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "Symbol" not in df.columns:
        raise ValueError("My-Stocks.csv must contain a column named 'Symbol'.")

    df["Symbol"] = (
        df["Symbol"].astype(str).str.strip().str.upper()
        .str.replace("\\", "", regex=False)
    )
    return df[df["Symbol"] != ""].drop_duplicates("Symbol").copy()


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


def get_candles(instrument_key, unit, interval, lookback_days):
    """
    Upstox Historical Candle V3.
    URL: /historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}
    """
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (
        datetime.now() - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    encoded_key = quote(instrument_key, safe="")

    response = requests.get(
        f"{UPSTOX_API}/v3/historical-candle/"
        f"{encoded_key}/{unit}/{interval}/{to_date}/{from_date}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(
            f"  {unit}/{interval} candle request failed: "
            f"{response.status_code} {response.text[:300]}"
        )
        return []

    return response.json().get("data", {}).get("candles", [])


def get_15m_candles(instrument_key):
    """Fetch 15-minute historical candles from Upstox V3."""
    return get_candles(
        instrument_key,
        "minutes",
        "15",
        FIFTEEN_MIN_LOOKBACK_DAYS,
    )


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
        get_candles(instrument_key, "weeks", "1", WEEKLY_LOOKBACK_DAYS)
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
        get_candles(instrument_key, "hours", "1", HOURLY_LOOKBACK_DAYS)
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


def get_completed_15m_rsi(instrument_key):
    """
    Calculate RSI using ONLY completed 15-minute candles.

    Reversal = latest completed 15m RSI is rising AND the
    preceding completed 15m RSI was not rising.
    """
    candles = get_15m_candles(instrument_key)

    if not candles:
        return None

    df = candle_dataframe(candles)

    if df.empty:
        return None

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    current_bucket = now.floor("15min")

    local_time = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    # Exclude the currently forming 15-minute candle.
    completed = df[local_time < current_bucket].copy()

    if len(completed) < RSI_PERIOD + 3:
        return None

    completed["RSI"] = calculate_rsi(
        completed["close"],
        RSI_PERIOD,
    )

    valid = completed.dropna(subset=["RSI"]).reset_index(drop=True)

    if len(valid) < 3:
        return None

    older = valid.iloc[-3]
    previous = valid.iloc[-2]
    latest = valid.iloc[-1]

    older_rsi = float(older["RSI"])
    previous_rsi = float(previous["RSI"])
    latest_rsi = float(latest["RSI"])

    previous_change = previous_rsi - older_rsi
    latest_change = latest_rsi - previous_rsi

    reversal = (
        latest_change > 0
        and previous_change <= 0
    )

    latest_time = latest["timestamp"].tz_convert("Asia/Kolkata")

    return {
        "candle": latest_time.strftime("%Y-%m-%d %H:%M"),
        "current": latest_rsi,
        "previous": previous_rsi,
        "older": older_rsi,
        "change": latest_change,
        "previous_change": previous_change,
        "rising": latest_change > 0,
        "reversal": reversal,
    }

def classify(weekly, hourly, fifteen):
    """
    Strategy:
      Weekly RSI > 50
      AND Hourly RSI < 30
      AND completed 15m RSI reversal.
    """
    weekly_rsi = weekly["current"]
    hourly_rsi = hourly["current"]

    if weekly_rsi <= 50:
        return "IGNORE", "❌ IGNORE", "Weekly RSI <= 50"

    if hourly_rsi >= 30:
        return (
            "WAIT",
            "⏳ WAIT",
            f"Hourly RSI {hourly_rsi:.2f} >= 30; waiting for oversold condition",
        )

    if fifteen is None:
        return "WAIT", "⏳ WAIT", "Completed 15m RSI unavailable"

    if fifteen["reversal"]:
        return (
            "SETUP",
            "🔥 SETUP",
            "Weekly RSI > 50 + Hourly RSI < 30 + completed 15m RSI reversal",
        )

    if fifteen["rising"]:
        return (
            "WATCH",
            "👀 WATCH",
            "Hourly RSI oversold + completed 15m RSI rising, but no fresh reversal",
        )

    return (
        "WATCH",
        "👀 WATCH",
        "Hourly RSI oversold + completed 15m RSI still falling",
    )


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
        "15m Completed Candle",
        "15m RSI",
        "15m Previous RSI",
        "15m Older RSI",
        "15m RSI Change",
        "15m Previous Change",
        "15m RSI Rising",
        "15m RSI Reversal",
        "Category",
        "Signal",
        "Reason",
    ]

    now_ist = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for result in results:
        rows.append({
            "Scan Time": now_ist,
            **{
                key: result.get(key, "")
                for key in history_columns
                if key != "Scan Time"
            },
        })

    new_history = pd.DataFrame(rows, columns=history_columns)

    new_history.to_csv(
        HISTORY_FILE,
        mode="a" if HISTORY_FILE.exists() else "w",
        header=not HISTORY_FILE.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def process_stock(symbol):
    print(f"\nChecking {symbol} ...")

    instrument_key = find_instrument_key(symbol)
    if not instrument_key:
        print("  ❌ Instrument not found")
        return None

    print(f"  Instrument: {instrument_key}")

    ltp = get_current_ltp(instrument_key)
    if ltp is None:
        print("  ❌ LTP unavailable")
        return None

    weekly = get_current_weekly_rsi(instrument_key, ltp)
    hourly = get_current_hourly_rsi(instrument_key, ltp)
    fifteen = get_completed_15m_rsi(instrument_key)

    if weekly is None or hourly is None or fifteen is None:
        print("  ❌ Could not calculate all RSI timeframes")
        return None

    category, signal, reason = classify(
        weekly,
        hourly,
        fifteen,
    )

    hourly_change = hourly["current"] - hourly["previous"]

    print(f"  LTP: ₹{ltp:.2f}")
    print(f"  Weekly RSI: {weekly['current']:.2f}")
    print(f"  Hourly RSI: {hourly['current']:.2f}")
    print(f"  Hourly Change: {hourly_change:+.2f}")
    print(f"  Completed 15m Candle: {fifteen['candle']}")
    print(
        f"  15m RSI: {fifteen['current']:.2f} | "
        f"Previous: {fifteen['previous']:.2f} | "
        f"Older: {fifteen['older']:.2f}"
    )
    print(
        f"  15m Change: {fifteen['change']:+.2f} | "
        f"Previous Change: {fifteen['previous_change']:+.2f}"
    )
    print(
        f"  15m Reversal: "
        f"{'YES' if fifteen['reversal'] else 'NO'}"
    )
    print(f"  Signal: {signal}")
    print(f"  Reason: {reason}")

    return {
        "Symbol": symbol,
        "Instrument Key": instrument_key,
        "Current LTP": round(ltp, 2),
        "Current Week RSI": round(weekly["current"], 2),
        "Previous Week RSI": round(weekly["previous"], 2),
        "Weekly RSI Change": round(
            weekly["current"] - weekly["previous"], 2
        ),
        "Current Hour": hourly["hour"],
        "Current Hourly RSI": round(hourly["current"], 2),
        "Previous Hourly RSI": round(hourly["previous"], 2),
        "Hourly RSI Change": round(hourly_change, 2),
        "Hourly RSI Rising": (
            "YES" if hourly_change > 0 else "NO"
        ),
        "15m Completed Candle": fifteen["candle"],
        "15m RSI": round(fifteen["current"], 2),
        "15m Previous RSI": round(fifteen["previous"], 2),
        "15m Older RSI": round(fifteen["older"], 2),
        "15m RSI Change": round(fifteen["change"], 2),
        "15m Previous Change": round(
            fifteen["previous_change"], 2
        ),
        "15m RSI Rising": (
            "YES" if fifteen["rising"] else "NO"
        ),
        "15m RSI Reversal": (
            "YES" if fifteen["reversal"] else "NO"
        ),
        "Category": category,
        "Signal": signal,
        "Reason": reason,
    }



def save_latest_results(output):
    """Write stable CSV and JSON files for the GitHub Pages dashboard."""
    output.to_csv(
        LATEST_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    output.to_json(
        LATEST_JSON_FILE,
        orient="records",
        date_format="iso",
        force_ascii=False,
        indent=2,
    )


def main():
    print("=" * 90)
    print("RSI SCANNER")
    print("Weekly RSI > 50 + Hourly RSI < 30 + Completed 15m RSI Reversal")
    print("=" * 90)

    stocks = load_stocks()
    print(f"Stocks found: {len(stocks)}")

    results = []

    for number, symbol in enumerate(stocks["Symbol"].tolist(), start=1):
        print(f"\n[{number}/{len(stocks)}] {symbol}")

        try:
            result = process_stock(symbol)
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

    # Timestamped archive report.
    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Stable files consumed by GitHub Pages.
    save_latest_results(output)

    # Persistent RSI history.
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
        "15m Completed Candle",
        "15m RSI",
        "15m Previous RSI",
        "15m Older RSI",
        "15m RSI Change",
        "15m Previous Change",
        "15m RSI Rising",
        "15m RSI Reversal",
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
    print(f"History saved/appended: {HISTORY_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
