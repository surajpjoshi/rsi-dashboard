import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# CONFIG
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_FOLDER / "My-Stocks.csv"

REPORT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = SCRIPT_FOLDER / f"RSI_Scanner_{REPORT_TIMESTAMP}.csv"

# Files consumed by GitHub Pages
LATEST_CSV_FILE = SCRIPT_FOLDER / "latest_results.csv"
LATEST_JSON_FILE = SCRIPT_FOLDER / "latest_results.json"

# Persistent history
HISTORY_FILE = SCRIPT_FOLDER / "RSI_History.csv"

RSI_PERIOD = 14
WEEKLY_LOOKBACK_DAYS = 730
HOURLY_LOOKBACK_DAYS = 30

UPSTOX_API = "https://api.upstox.com"

ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()

if not ACCESS_TOKEN:
    raise SystemExit("ERROR: UPSTOX_ACCESS_TOKEN is not configured.")

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}


# ============================================================
# RSI
# ============================================================

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
        rsi.iloc[period] = 100 - (
            100 / (1 + avg_gain / avg_loss)
        )

    for i in range(period + 1, len(close)):
        avg_gain = (
            (avg_gain * (period - 1)) + gain.iloc[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + loss.iloc[i]
        ) / period

        if avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rsi.iloc[i] = 100 - (
                100 / (1 + avg_gain / avg_loss)
            )

    return rsi


# ============================================================
# STOCK LIST
# ============================================================

def load_stocks():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"My-Stocks.csv not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if "Symbol" not in df.columns:
        raise ValueError(
            "My-Stocks.csv must contain a column named 'Symbol'."
        )

    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("\\", "", regex=False)
    )

    return (
        df[df["Symbol"] != ""]
        .drop_duplicates("Symbol")
        .copy()
    )


# ============================================================
# UPSTOX
# ============================================================

def find_instrument_key(symbol):
    trading_symbol = (
        symbol.replace("NSE:", "")
        .strip()
        .upper()
    )

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
        print(
            f"  Instrument search failed: "
            f"{response.status_code}"
        )
        return None

    instruments = response.json().get("data", [])

    for item in instruments:
        if (
            str(item.get("trading_symbol", "")).upper()
            == trading_symbol
            and item.get("segment") == "NSE_EQ"
        ):
            return item.get("instrument_key")

    for item in instruments:
        if item.get("segment") == "NSE_EQ":
            return item.get("instrument_key")

    return None


def get_candles(
    instrument_key,
    unit,
    interval,
    lookback_days
):
    """
    Upstox Historical Candle V3.
    """

    to_date = datetime.now().strftime("%Y-%m-%d")

    from_date = (
        datetime.now()
        - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    encoded_key = quote(instrument_key, safe="")

    response = requests.get(
        f"{UPSTOX_API}/v3/historical-candle/"
        f"{encoded_key}/{unit}/{interval}/"
        f"{to_date}/{from_date}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(
            f"  {unit}/{interval} candle request failed: "
            f"{response.status_code} "
            f"{response.text[:300]}"
        )
        return []

    return (
        response.json()
        .get("data", {})
        .get("candles", [])
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
        print(
            f"  LTP request failed: "
            f"{response.status_code}"
        )
        return None

    quote_data = response.json().get("data", {})

    if not quote_data:
        return None

    first_quote = next(iter(quote_data.values()))

    last_price = first_quote.get("last_price")

    return (
        float(last_price)
        if last_price is not None
        else None
    )


# ============================================================
# CANDLE DATAFRAME
# ============================================================

def candle_dataframe(candles):
    rows = [
        {
            "timestamp": c[0],
            "close": c[4],
        }
        for c in candles
        if len(c) >= 5
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    return (
        df.dropna(
            subset=["timestamp", "close"]
        )
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# WEEKLY RSI
# ============================================================

def get_current_weekly_rsi(
    instrument_key,
    ltp
):
    df = candle_dataframe(
        get_candles(
            instrument_key,
            "weeks",
            "1",
            WEEKLY_LOOKBACK_DAYS,
        )
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    week_start = (
        now
        - pd.Timedelta(days=now.weekday())
    ).normalize()

    local_time = (
        df["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
    )

    completed = df[
        local_time < week_start
    ].copy()

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([
        {
            "timestamp": week_start,
            "close": ltp,
        }
    ])

    calc = pd.concat(
        [
            completed[["timestamp", "close"]],
            current,
        ],
        ignore_index=True,
    )

    calc["RSI"] = calculate_rsi(
        calc["close"],
        RSI_PERIOD,
    )

    valid = calc.dropna(
        subset=["RSI"]
    )

    if len(valid) < 2:
        return None

    return {
        "current": float(
            valid.iloc[-1]["RSI"]
        ),
        "previous": float(
            valid.iloc[-2]["RSI"]
        ),
    }


# ============================================================
# HOURLY RSI
# ============================================================

def get_current_hourly_rsi(
    instrument_key,
    ltp
):
    df = candle_dataframe(
        get_candles(
            instrument_key,
            "hours",
            "1",
            HOURLY_LOOKBACK_DAYS,
        )
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    current_hour = now.floor("h")

    local_time = (
        df["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
    )

    completed = df[
        local_time < current_hour
    ].copy()

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([
        {
            "timestamp": current_hour,
            "close": ltp,
        }
    ])

    calc = pd.concat(
        [
            completed[["timestamp", "close"]],
            current,
        ],
        ignore_index=True,
    )

    calc["RSI"] = calculate_rsi(
        calc["close"],
        RSI_PERIOD,
    )

    valid = calc.dropna(
        subset=["RSI"]
    )

    if len(valid) < 2:
        return None

    return {
        "current": float(
            valid.iloc[-1]["RSI"]
        ),
        "previous": float(
            valid.iloc[-2]["RSI"]
        ),
        "hour": current_hour.strftime(
            "%Y-%m-%d %H:%M"
        ),
    }


# ============================================================
# SIGNAL LOGIC
# ============================================================

def classify(weekly, hourly):
    """
    ONLY Weekly + Hourly logic.

    Weekly RSI > 50
        +
    Hourly RSI < 30  = SETUP

    Hourly RSI 30-50 = WATCH
    Hourly RSI >= 50 = WAIT
    Weekly RSI <= 50 = IGNORE
    """

    weekly_rsi = float(
        weekly["current"]
    )

    hourly_rsi = float(
        hourly["current"]
    )

    if weekly_rsi <= 50:
        return (
            "IGNORE",
            "❌ IGNORE",
            f"Weekly RSI {weekly_rsi:.2f} <= 50",
        )

    if hourly_rsi < 30:
        return (
            "SETUP",
            "🔥 SETUP",
            f"Weekly RSI {weekly_rsi:.2f} > 50 + "
            f"Hourly RSI {hourly_rsi:.2f} < 30",
        )

    if hourly_rsi < 50:
        return (
            "WATCH",
            "👀 WATCH",
            f"Hourly RSI {hourly_rsi:.2f} is 30-50",
        )

    return (
        "WAIT",
        "⏳ WAIT",
        f"Hourly RSI {hourly_rsi:.2f} >= 50",
    )


# ============================================================
# PROCESS STOCK
# ============================================================

def process_stock(symbol):
    print(f"\nChecking {symbol} ...")

    instrument_key = find_instrument_key(symbol)

    if not instrument_key:
        print("  ❌ Instrument not found")
        return None

    print(
        f"  Instrument: {instrument_key}"
    )

    ltp = get_current_ltp(
        instrument_key
    )

    if ltp is None:
        print("  ❌ LTP unavailable")
        return None

    weekly = get_current_weekly_rsi(
        instrument_key,
        ltp,
    )

    hourly = get_current_hourly_rsi(
        instrument_key,
        ltp,
    )

    if weekly is None or hourly is None:
        print(
            "  ❌ Could not calculate "
            "Weekly/Hourly RSI"
        )
        return None

    category, signal, reason = classify(
        weekly,
        hourly,
    )

    hourly_change = (
        hourly["current"]
        - hourly["previous"]
    )

    scan_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"  LTP: ₹{ltp:.2f}"
    )

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
        "  15m logic: REMOVED"
    )

    print(
        f"  Signal: {signal}"
    )

    print(
        f"  Reason: {reason}"
    )

    return {
        "Scan Time": scan_time,
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
            weekly["current"]
            - weekly["previous"],
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

        # Kept blank for backward compatibility
        # with the existing dashboard/history files.
        "Current 15m Candle": "",
        "Current 15m RSI": "",
        "Previous 15m Candle RSI": "",
        "15m RSI Change": "",
        "15m RSI Rising": "",
        "15m Rising Count": "",
        "History Transition": (
            "15m logic removed"
        ),

        "Category": category,
        "Signal": signal,
        "Reason": reason,
    }


# ============================================================
# LATEST RESULTS
# ============================================================

def save_latest_results(output):
    """
    These two files are the ONLY stable files
    required by GitHub Pages.
    """

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

    print(
        f"Latest CSV saved: "
        f"{LATEST_CSV_FILE}"
    )

    print(
        f"Latest JSON saved: "
        f"{LATEST_JSON_FILE}"
    )


# ============================================================
# RSI HISTORY
# ============================================================

HISTORY_COLUMNS = [
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


def save_rsi_history(results):
    """
    Append one row per stock to RSI_History.csv.

    This function is intentionally independent from
    the dashboard output so a history problem cannot
    prevent latest_results.json from being generated.
    """

    history_rows = []

    for item in results:
        row = {
            column: item.get(
                column,
                "",
            )
            for column in HISTORY_COLUMNS
        }

        history_rows.append(row)

    new_history = pd.DataFrame(
        history_rows,
        columns=HISTORY_COLUMNS,
    )

    if HISTORY_FILE.exists():
        try:
            existing = pd.read_csv(
                HISTORY_FILE,
                encoding="utf-8-sig",
            )

            # Preserve older history while normalizing
            # it to the current column structure.
            existing = existing.reindex(
                columns=HISTORY_COLUMNS,
                fill_value="",
            )

            combined = pd.concat(
                [existing, new_history],
                ignore_index=True,
            )

        except Exception as error:
            print(
                "  ⚠️ Could not read existing "
                f"RSI_History.csv: {error}"
            )
            combined = new_history

    else:
        combined = new_history

    combined.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"History saved/appended: "
        f"{HISTORY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("RSI SCANNER")
    print("Weekly RSI > 50 + Hourly RSI < 30")
    print("15-minute logic REMOVED")
    print("=" * 90)

    stocks = load_stocks()

    print(
        f"Stocks found: {len(stocks)}"
    )

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
            result = process_stock(
                symbol
            )

            if result is not None:
                results.append(result)

        except Exception as error:
            print(
                f"  ❌ ERROR: {error}"
            )

        time.sleep(0.3)

    if not results:
        print(
            "\n❌ No results generated."
        )
        return 1

    output = pd.DataFrame(results)

    priority = {
        "SETUP": 1,
        "WATCH": 2,
        "WAIT": 3,
        "IGNORE": 4,
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
                "Current Hourly RSI",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .drop(
            columns=["_priority"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Save dashboard files BEFORE history.
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nReport saved: "
        f"{OUTPUT_FILE}"
    )

    save_latest_results(
        output
    )

    # History is now safe because latest_results
    # has already been written.
    try:
        save_rsi_history(
            results
        )
    except Exception as error:
        print(
            "  ⚠️ RSI history update failed, "
            f"but dashboard files are ready: {error}"
        )

    setup_count = int(
        (
            output["Category"]
            == "SETUP"
        ).sum()
    )

    watch_count = int(
        (
            output["Category"]
            == "WATCH"
        ).sum()
    )

    wait_count = int(
        (
            output["Category"]
            == "WAIT"
        ).sum()
    )

    ignore_count = int(
        (
            output["Category"]
            == "IGNORE"
        ).sum()
    )

    print("\n" + "=" * 90)
    print("FINAL RSI SCANNER RESULT")
    print("=" * 90)

    columns = [
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
        "Category",
        "Signal",
        "Reason",
    ]

    print(
        output[columns]
        .to_string(index=False)
    )

    print("\n" + "=" * 90)
    print(
        f"🔥 SETUP : {setup_count}"
    )
    print(
        f"👀 WATCH : {watch_count}"
    )
    print(
        f"⏳ WAIT  : {wait_count}"
    )
    print(
        f"❌ IGNORE: {ignore_count}"
    )

    print(
        f"\nReport saved: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Latest JSON: "
        f"{LATEST_JSON_FILE}"
    )

    print(
        f"History saved/appended: "
        f"{HISTORY_FILE}"
    )

    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
