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
MINUTE15_LOOKBACK_DAYS = 15

UPSTOX_API = "https://api.upstox.com"

# Stable files used by GitHub Pages / GitHub Actions.
LATEST_OUTPUT_FILE = SCRIPT_FOLDER / "latest_results.csv"
LATEST_JSON_FILE = SCRIPT_FOLDER / "latest_results.json"

# GitHub Actions provides the token through the UPSTOX_ACCESS_TOKEN secret.
# For local Windows execution you can also set the same environment variable.
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


def get_candles(instrument_key, unit, lookback_days, interval=1):
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
            f"{response.status_code}"
        )
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


def get_current_15m_rsi(instrument_key, ltp):
    """
    Calculate RSI(14) from 15-minute candles.

    We use completed 15-minute candles and append the current
    15-minute bucket using the live LTP, so the value can change
    on every 15-minute GitHub scan.
    """
    df = candle_dataframe(
        get_candles(
            instrument_key,
            "minutes",
            MINUTE15_LOOKBACK_DAYS,
            interval=15,
        )
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    current_15m = now.floor("15min")

    local_time = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    # Do not use the current live candle from the API if one exists;
    # we rebuild the current bucket using the latest LTP.
    completed = df[local_time < current_15m].copy()

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([{
        "timestamp": current_15m,
        "close": ltp,
    }])

    calc = pd.concat(
        [completed[["timestamp", "close"]], current],
        ignore_index=True,
    )

    calc["RSI"] = calculate_rsi(calc["close"], RSI_PERIOD)
    valid = calc.dropna(subset=["RSI"])

    if len(valid) < 2:
        return None

    return {
        "current": float(valid.iloc[-1]["RSI"]),
        "previous": float(valid.iloc[-2]["RSI"]),
        "candle": current_15m.strftime("%Y-%m-%d %H:%M"),
    }


def get_previous_scan_for_symbol(symbol):
    """
    Return the latest saved scanner row for a symbol.
    """
    if not HISTORY_FILE.exists():
        return None

    try:
        history = pd.read_csv(HISTORY_FILE)
    except Exception:
        return None

    if history.empty or "Symbol" not in history.columns:
        return None

    rows = history[
        history["Symbol"].astype(str).str.upper() == symbol.upper()
    ].copy()

    if rows.empty:
        return None

    if "Scan Time" in rows.columns:
        rows["_sort_time"] = pd.to_datetime(
            rows["Scan Time"],
            errors="coerce",
        )
        rows = rows.sort_values("_sort_time")

    return rows.iloc[-1].to_dict()


def get_15m_rising_count(symbol, current_15m_rsi):
    """
    Count consecutive 15-minute RSI rises across scanner runs.

    Example:
      24.0 -> 24.8 -> 26.1 -> 27.4
      rising count = 3

    The count resets to 0 when the current scanner RSI is flat/falling.
    """
    if not HISTORY_FILE.exists():
        return 0

    try:
        history = pd.read_csv(HISTORY_FILE)
    except Exception:
        return 0

    required = {"Symbol", "Scan Time", "Current 15m RSI"}
    if not required.issubset(history.columns):
        return 0

    rows = history[
        history["Symbol"].astype(str).str.upper() == symbol.upper()
    ].copy()

    if rows.empty:
        return 0

    rows["_sort_time"] = pd.to_datetime(
        rows["Scan Time"],
        errors="coerce",
    )
    rows["Current 15m RSI"] = pd.to_numeric(
        rows["Current 15m RSI"],
        errors="coerce",
    )
    rows = rows.dropna(
        subset=["_sort_time", "Current 15m RSI"]
    ).sort_values("_sort_time")

    if rows.empty:
        return 0

    values = rows["Current 15m RSI"].tolist()

    count = 0

    # Walk backwards through consecutive historical increases.
    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break

    if count == 0 and values:
        # Compare current live value with the last saved value.
        if current_15m_rsi > values[-1]:
            count = 1

    return count


def classify(weekly, hourly, rsi15, rising_count):
    """
    Trading logic:

    1. Weekly RSI must be > 50.
    2. Hourly RSI must be <= 30 for the reversal setup.
    3. 15-minute RSI must start rising.
    4. Two or more consecutive 15-minute rises = SETUP.
    """
    weekly_rsi = weekly["current"]
    hourly_rsi = hourly["current"]
    rsi15_current = rsi15["current"]

    previous_15m = rsi15["previous"]
    rsi15_rising = rsi15_current > previous_15m

    if weekly_rsi <= 50:
        return (
            "IGNORE",
            "❌ IGNORE",
            "Weekly RSI <= 50",
        )

    # Strong trend, but not an oversold entry.
    if hourly_rsi > 50:
        return (
            "WAIT",
            "⏳ WAIT",
            "Hourly RSI > 50",
        )

    # Pullback zone.
    if 30 <= hourly_rsi <= 50:
        if rsi15_rising:
            return (
                "WATCH",
                "👀 WATCH",
                "Hourly RSI 30-50 + 15m RSI rising",
            )

        return (
            "WATCH",
            "👀 WATCH",
            "Hourly RSI 30-50",
        )

    # Oversold zone.
    if hourly_rsi <= 30:

        if rsi15_rising and rising_count >= 2:
            return (
                "SETUP",
                "🔥 SETUP",
                "Weekly RSI > 50 + Hourly RSI <= 30 + "
                "15m RSI rising for >= 2 scans",
            )

        if rsi15_rising and rising_count == 1:
            return (
                "NEAR SETUP",
                "🟡 NEAR SETUP",
                "Hourly RSI <= 30 + first 15m RSI rise",
            )

        return (
            "WATCH",
            "👀 WATCH",
            "Hourly RSI <= 30 but 15m RSI is not rising",
        )

    return (
        "WAIT",
        "⏳ WAIT",
        "No setup",
    )


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
        try:
            existing = pd.read_csv(HISTORY_FILE)

            # Migrate the existing GitHub history file to the new schema.
            # Older columns are preserved where possible.
            for column in history_columns:
                if column not in existing.columns:
                    existing[column] = ""

            existing = existing[
                [c for c in history_columns if c in existing.columns]
            ]

            combined = pd.concat(
                [existing, new_history],
                ignore_index=True,
            )

            combined.to_csv(
                HISTORY_FILE,
                index=False,
                encoding="utf-8-sig",
            )

        except Exception as error:
            print(
                f"  History migration/read failed: {error}"
            )

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


def detect_history_transition(
    symbol,
    current_hourly_rsi,
    current_weekly_rsi,
    current_15m_rsi,
):
    """
    Describe the current hourly/15m state for the report.
    """

    previous = get_previous_scan_for_symbol(symbol)

    if previous is None:
        return "FIRST SCAN"

    previous_15m = pd.to_numeric(
        previous.get("Current 15m RSI"),
        errors="coerce",
    )

    if pd.isna(previous_15m):
        return "NO PREVIOUS 15m RSI"

    change = current_15m_rsi - previous_15m

    if current_weekly_rsi <= 50:
        return "WEEKLY RSI <= 50"

    if current_hourly_rsi <= 30:
        if change > 0:
            return "OVERSOLD + 15m RISING"
        if change < 0:
            return "OVERSOLD + 15m FALLING"
        return "OVERSOLD + 15m FLAT"

    if 30 <= current_hourly_rsi <= 50:
        if change > 0:
            return "30-50 + 15m RISING"
        if change < 0:
            return "30-50 + 15m FALLING"
        return "30-50 + 15m FLAT"

    if change > 0:
        return "ABOVE 50 + 15m RISING"
    if change < 0:
        return "ABOVE 50 + 15m FALLING"

    return "ABOVE 50 + 15m FLAT"


def process_stock(symbol):
    print(f"\nChecking {symbol} ...")

    instrument_key = find_instrument_key(symbol)

    if not instrument_key:
        return None

    ltp = get_current_ltp(instrument_key)

    if ltp is None:
        return None

    weekly = get_current_weekly_rsi(
        instrument_key,
        ltp,
    )

    hourly = get_current_hourly_rsi(
        instrument_key,
        ltp,
    )

    rsi15 = get_current_15m_rsi(
        instrument_key,
        ltp,
    )

    if (
        weekly is None
        or hourly is None
        or rsi15 is None
    ):
        return None

    previous_scan = get_previous_scan_for_symbol(symbol)

    previous_scan_15m = None

    if previous_scan is not None:
        previous_scan_15m = pd.to_numeric(
            previous_scan.get("Current 15m RSI"),
            errors="coerce",
        )

        if pd.isna(previous_scan_15m):
            previous_scan_15m = None

    # Compare the current 15m RSI with the previous scanner run.
    if previous_scan_15m is not None:
        scan_15m_change = (
            rsi15["current"] - previous_scan_15m
        )
    else:
        scan_15m_change = (
            rsi15["current"] - rsi15["previous"]
        )

    scan_15m_rising = scan_15m_change > 0

    # Calculate consecutive scanner-level rises.
    if previous_scan_15m is not None:
        rising_count = (
            get_15m_rising_count(
                symbol,
                rsi15["current"],
            )
            + (1 if scan_15m_rising else 0)
        )
    else:
        rising_count = 1 if scan_15m_rising else 0

    category, signal, reason = classify(
        weekly,
        hourly,
        rsi15,
        rising_count,
    )

    hourly_change = (
        hourly["current"] -
        hourly["previous"]
    )

    history_transition = detect_history_transition(
        symbol,
        hourly["current"],
        weekly["current"],
        rsi15["current"],
    )

    print(f"  LTP: ₹{ltp:.2f}")
    print(
        f"  Weekly RSI: "
        f"{weekly['current']:.2f}"
    )
    print(
        f"  Hourly RSI: "
        f"{hourly['current']:.2f}"
    )
    print(
        f"  Hourly Change: "
        f"{hourly_change:+.2f}"
    )
    print(
        f"  15m RSI: "
        f"{rsi15['current']:.2f}"
    )
    print(
        f"  15m Scan Change: "
        f"{scan_15m_change:+.2f}"
    )
    print(
        f"  15m Rising: "
        f"{'YES' if scan_15m_rising else 'NO'}"
    )
    print(
        f"  15m Rising Count: "
        f"{rising_count}"
    )
    print(
        f"  History Transition: "
        f"{history_transition}"
    )
    print(f"  Signal: {signal}")

    return {
        "Symbol": symbol,
        "Instrument Key": instrument_key,
        "Current LTP": round(ltp, 2),

        "Current Week RSI": round(
            weekly["current"],
            2,
        ),
        "Previous Week RSI": round(
            weekly["previous"],
            2,
        ),
        "Weekly RSI Change": round(
            weekly["current"] -
            weekly["previous"],
            2,
        ),

        "Current Hour": hourly["hour"],
        "Current Hourly RSI": round(
            hourly["current"],
            2,
        ),
        "Previous Hourly RSI": round(
            hourly["previous"],
            2,
        ),
        "Hourly RSI Change": round(
            hourly_change,
            2,
        ),
        "Hourly RSI Rising": (
            "YES"
            if hourly_change > 0
            else "NO"
        ),

        "Current 15m Candle": rsi15["candle"],
        "Current 15m RSI": round(
            rsi15["current"],
            2,
        ),
        "Previous 15m Candle RSI": round(
            rsi15["previous"],
            2,
        ),
        "15m RSI Change": round(
            scan_15m_change,
            2,
        ),
        "15m RSI Rising": (
            "YES"
            if scan_15m_rising
            else "NO"
        ),
        "15m Rising Count": rising_count,

        "History Transition": history_transition,
        "Category": category,
        "Signal": signal,
        "Reason": reason,
    }


def main():
    print("=" * 90)
    print("RSI SCANNER")
    print(
        "Weekly RSI > 50 + Hourly RSI <= 30 "
        "+ 15m RSI Reversal"
    )
    print("=" * 90)

    stocks = load_stocks()

    print(f"Stocks found: {len(stocks)}")

    results = []

    for number, symbol in enumerate(
        stocks["Symbol"].tolist(),
        start=1,
    ):
        print(
            f"\n[{number}/{len(stocks)}] "
            f"{symbol}"
        )

        try:
            result = process_stock(symbol)

            if result is not None:
                results.append(result)

        except Exception as error:
            print(
                f"  ERROR: {error}"
            )

        time.sleep(0.3)

    if not results:
        print(
            "\nNo results generated."
        )
        return

    output = pd.DataFrame(results)

    priority = {
        "SETUP": 1,
        "NEAR SETUP": 2,
        "WATCH": 3,
        "WAIT": 4,
        "IGNORE": 5,
    }

    output["_priority"] = (
        output["Category"]
        .map(priority)
        .fillna(99)
    )

    output = (
        output.sort_values(
            [
                "_priority",
                "15m Rising Count",
                "Current Hourly RSI",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop(
            columns=["_priority"]
        )
        .reset_index(drop=True)
    )

    # Timestamped archive.
    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Stable latest CSV for GitHub Pages.
    output.to_csv(
        LATEST_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Stable JSON for GitHub Pages.
    output.to_json(
        LATEST_JSON_FILE,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # Persist this scan for future comparisons.
    save_rsi_history(results)

    setup_count = int(
        (output["Category"] == "SETUP").sum()
    )

    near_setup_count = int(
        (output["Category"] == "NEAR SETUP").sum()
    )

    watch_count = int(
        (output["Category"] == "WATCH").sum()
    )

    wait_count = int(
        (output["Category"] == "WAIT").sum()
    )

    ignore_count = int(
        (output["Category"] == "IGNORE").sum()
    )

    print(
        "\n" + "=" * 90
    )
    print(
        "FINAL RSI SCANNER RESULT"
    )
    print(
        "=" * 90
    )

    columns = [
        "Symbol",
        "Current Week RSI",
        "Current Hourly RSI",
        "Current 15m RSI",
        "15m RSI Change",
        "15m RSI Rising",
        "15m Rising Count",
        "Category",
        "Signal",
        "Reason",
    ]

    print(
        output[columns].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 90
    )

    print(
        f"🔥 SETUP      : "
        f"{setup_count}"
    )

    print(
        f"🟡 NEAR SETUP : "
        f"{near_setup_count}"
    )

    print(
        f"👀 WATCH      : "
        f"{watch_count}"
    )

    print(
        f"⏳ WAIT       : "
        f"{wait_count}"
    )

    print(
        f"❌ IGNORE     : "
        f"{ignore_count}"
    )

    print(
        f"\nReport saved: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Latest report: "
        f"{LATEST_OUTPUT_FILE}"
    )

    print(
        f"Latest JSON: "
        f"{LATEST_JSON_FILE}"
    )

    print(
        f"History saved/appended: "
        f"{HISTORY_FILE}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
