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

# Persistent RSI history. Every scan appends one row per stock.
HISTORY_FILE = SCRIPT_FOLDER / "RSI_History.csv"

RSI_PERIOD = 14
WEEKLY_LOOKBACK_DAYS = 730
HOURLY_LOOKBACK_DAYS = 30
MIN15_LOOKBACK_DAYS = 15
UPSTOX_API = "https://api.upstox.com"

# Upstox token:
# GitHub Actions supplies UPSTOX_ACCESS_TOKEN as a repository secret.
# For local Windows testing, set the environment variable first.
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


def get_completed_15m_rsi(instrument_key):
    """
    Calculate RSI(14) using ONLY completed 15-minute candles.

    The currently forming 15-minute candle is excluded:
        current_time = floor(now, 15 minutes)
        completed candles = candle timestamp < current_time

    This prevents the RSI from changing during a live 15-minute candle
    and prevents repeated scanner runs from counting the same candle twice.
    """
    df = candle_dataframe(
        get_candles(instrument_key, "minutes/15", MIN15_LOOKBACK_DAYS)
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    current_15m_start = now.floor("15min")

    local_time = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    # IMPORTANT: only completed 15-minute candles.
    completed = df[local_time < current_15m_start].copy()

    if len(completed) < RSI_PERIOD + 3:
        return None

    completed["RSI"] = calculate_rsi(
        completed["close"],
        RSI_PERIOD,
    )

    valid = completed.dropna(subset=["RSI"]).copy()

    if len(valid) < 3:
        return None

    # Current completed candle and previous completed candle.
    current_row = valid.iloc[-1]
    previous_row = valid.iloc[-2]

    current_rsi = float(current_row["RSI"])
    previous_rsi = float(previous_row["RSI"])

    # Count consecutive rising 15-minute candles ending at current candle.
    rising_count = 0

    for i in range(len(valid) - 1, 0, -1):
        current_value = float(valid.iloc[i]["RSI"])
        previous_value = float(valid.iloc[i - 1]["RSI"])

        if current_value > previous_value:
            rising_count += 1
        else:
            break

    change = current_rsi - previous_rsi

    return {
        "current": current_rsi,
        "previous": previous_rsi,
        "change": change,
        "rising": change > 0,
        "rising_count": rising_count,
        "candle": current_row["timestamp"].tz_convert(
            "Asia/Kolkata"
        ).strftime("%Y-%m-%d %H:%M"),
        "previous_candle": previous_row["timestamp"].tz_convert(
            "Asia/Kolkata"
        ).strftime("%Y-%m-%d %H:%M"),
    }


def classify(weekly, hourly, min15):
    """
    Signal logic:

    1. Weekly RSI > 50
    2. Hourly RSI is the pullback/oversold filter
    3. If Hourly RSI < 30:
         - 15m rising count >= 2 -> SETUP
         - 15m rising count == 1 -> NEAR SETUP
         - otherwise -> WATCH
    4. Hourly RSI 30-50 -> WATCH
    5. Hourly RSI > 50 -> WAIT
    """
    weekly_rsi = weekly["current"]
    hourly_rsi = hourly["current"]

    if weekly_rsi <= 50:
        return "IGNORE", "❌ IGNORE", "Weekly RSI <= 50"

    if hourly_rsi > 50:
        return "WAIT", "⏳ WAIT", "Hourly RSI > 50"

    if 30 <= hourly_rsi <= 50:
        if min15["rising"]:
            return (
                "WATCH",
                "👀 WATCH",
                "Hourly RSI 30-50; 15m RSI rising"
            )
        return (
            "WATCH",
            "👀 WATCH",
            "Hourly RSI 30-50"
        )

    # Hourly RSI < 30 = oversold.
    if hourly_rsi < 30:
        if min15["rising_count"] >= 2:
            return (
                "SETUP",
                "🔥 SETUP",
                "Weekly RSI > 50 + Hourly RSI < 30 + "
                "15m RSI rising for 2+ completed candles",
            )

        if min15["rising_count"] == 1:
            return (
                "NEAR SETUP",
                "🟡 NEAR SETUP",
                "Hourly RSI < 30 + 15m RSI has started rising",
            )

        return (
            "WATCH",
            "👀 WATCH",
            "Hourly RSI < 30 but completed 15m RSI is not rising",
        )

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
        "Current 15m Candle",
        "Current 15m RSI",
        "Previous 15m Candle RSI",
        "15m RSI Change",
        "15m RSI Rising",
        "15m Rising Count",
        "History Transition",
        "Category",
        "Signal",
        "Reason",
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
            "Current 15m Candle": result["Current 15m Candle"],
            "Current 15m RSI": result["Current 15m RSI"],
            "Previous 15m Candle RSI": result["Previous 15m Candle RSI"],
            "15m RSI Change": result["15m RSI Change"],
            "15m RSI Rising": result["15m RSI Rising"],
            "15m Rising Count": result["15m Rising Count"],
            "History Transition": result["History Transition"],
            "Category": result["Category"],
            "Signal": result["Signal"],
            "Reason": result["Reason"],
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


def get_history_transition(weekly, hourly, min15):
    """Create a human-readable transition for the current completed 15m candle."""
    if weekly["current"] <= 50:
        return "WEEKLY RSI <= 50"

    if hourly["current"] > 50:
        return "HOURLY RSI > 50"

    if hourly["current"] >= 30:
        return (
            "HOURLY RSI 30-50 + "
            + ("15m RISING" if min15["rising"] else "15m FALLING")
        )

    if min15["rising_count"] >= 2:
        return "OVERSOLD + 15m RISING 2+"

    if min15["rising_count"] == 1:
        return "OVERSOLD + 15m FIRST RISE"

    return "OVERSOLD + 15m FALLING"


def process_stock(symbol):
    print(f"\nChecking {symbol} ...")

    instrument_key = find_instrument_key(symbol)
    if not instrument_key:
        return None

    ltp = get_current_ltp(instrument_key)
    if ltp is None:
        return None

    weekly = get_current_weekly_rsi(instrument_key, ltp)
    hourly = get_current_hourly_rsi(instrument_key, ltp)
    min15 = get_completed_15m_rsi(instrument_key)

    if weekly is None or hourly is None or min15 is None:
        print("  ERROR: Could not calculate all RSI timeframes")
        return None

    category, signal, reason = classify(
        weekly,
        hourly,
        min15,
    )

    hourly_change = hourly["current"] - hourly["previous"]
    history_transition = get_history_transition(
        weekly,
        hourly,
        min15,
    )

    print(f"  LTP: ₹{ltp:.2f}")
    print(f"  Weekly RSI: {weekly['current']:.2f}")
    print(f"  Hourly RSI: {hourly['current']:.2f}")
    print(f"  Hourly Change: {hourly_change:+.2f}")
    print(
        f"  Completed 15m Candle: {min15['candle']}"
    )
    print(
        f"  15m RSI: {min15['current']:.2f}"
        f" | Previous: {min15['previous']:.2f}"
        f" | Change: {min15['change']:+.2f}"
    )
    print(
        f"  15m Rising: "
        f"{'YES' if min15['rising'] else 'NO'}"
        f" | Rising Count: {min15['rising_count']}"
    )
    print(f"  History Transition: {history_transition}")
    print(f"  Signal: {signal}")

    return {
        "Symbol": symbol,
        "Instrument Key": instrument_key,
        "Current LTP": round(ltp, 2),
        "Current Week RSI": round(weekly["current"], 2),
        "Previous Week RSI": round(weekly["previous"], 2),
        "Weekly RSI Change": round(
            weekly["current"] - weekly["previous"],
            2,
        ),
        "Current Hour": hourly["hour"],
        "Current Hourly RSI": round(hourly["current"], 2),
        "Previous Hourly RSI": round(hourly["previous"], 2),
        "Hourly RSI Change": round(hourly_change, 2),
        "Hourly RSI Rising": (
            "YES" if hourly_change > 0 else "NO"
        ),
        "Current 15m Candle": min15["candle"],
        "Current 15m RSI": round(min15["current"], 2),
        "Previous 15m Candle RSI": round(min15["previous"], 2),
        "15m RSI Change": round(min15["change"], 2),
        "15m RSI Rising": (
            "YES" if min15["rising"] else "NO"
        ),
        "15m Rising Count": int(min15["rising_count"]),
        "History Transition": history_transition,
        "Category": category,
        "Signal": signal,
        "Reason": reason,
    }


def main():
    print("=" * 90)
    print("RSI SCANNER")
    print("Weekly RSI > 50 + Hourly RSI Oversold + Completed 15m RSI Reversal")
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

    priority = {"SETUP": 1, "NEAR SETUP": 2, "WATCH": 3, "WAIT": 4, "IGNORE": 5}
    output["_priority"] = output["Category"].map(priority).fillna(99)

    output = (
        output.sort_values(
            ["_priority", "Current Hourly RSI"],
            ascending=[True, True],
        )
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Persist this scan for future comparisons.
    save_rsi_history(results)

    setup_count = int((output["Category"] == "SETUP").sum())
    near_setup_count = int((output["Category"] == "NEAR SETUP").sum())
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
    print(f"🔥 SETUP     : {setup_count}")
    print(f"🟡 NEAR SETUP: {near_setup_count}")
    print(f"👀 WATCH     : {watch_count}")
    print(f"⏳ WAIT  : {wait_count}")
    print(f"❌ IGNORE: {ignore_count}")
    print(f"\nReport saved: {OUTPUT_FILE}")
    print(f"History saved/appended: {HISTORY_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
