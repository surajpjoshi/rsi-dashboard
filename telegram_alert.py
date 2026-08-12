import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

DATA_FILE = Path("latest_results.json")
STATE_FILE = Path("telegram_alert_state.json")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def telegram_send(message):
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID GitHub secret."
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    request = urllib.request.Request(url, data=payload, method="POST")

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


def load_json(path, default):
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean(value):
    if value is None:
        return "--"

    text = str(value)

    # Telegram HTML safety
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value):
    n = num(value)

    if n is None:
        return clean(value)

    return f"{n:.2f}"


def delta(value):
    n = num(value)

    if n is None:
        return "--"

    return f"{n:+.2f}"


def signal_icon(category):
    category = str(category or "").upper()

    if category == "SETUP":
        return "🔥"

    if category == "WATCH":
        return "👀"

    if category == "WAIT":
        return "⏳"

    if category == "IGNORE":
        return "❌"

    return "📊"


def build_message(row, event):
    symbol = clean(row.get("Symbol"))
    category = str(row.get("Category") or "").upper()

    icon = signal_icon(category)

    if event == "NEW_SETUP":
        title = "🔥 <b>NEW RSI SETUP</b>"
    elif event == "NEW_WATCH":
        title = "👀 <b>NEW RSI WATCH</b>"
    elif event == "SETUP_LOST":
        title = "⚠️ <b>RSI SETUP LOST</b>"
    else:
        title = f"{icon} <b>RSI SIGNAL</b>"

    return f"""
{title}

<b>{symbol}</b>

💰 LTP: ₹{fmt(row.get("Current LTP"))}

📅 Weekly RSI: <b>{fmt(row.get("Current Week RSI"))}</b>
   Weekly Δ: {delta(row.get("Weekly RSI Change"))}

⏱ Hourly RSI: <b>{fmt(row.get("Current Hourly RSI"))}</b>
   Hourly Δ: {delta(row.get("Hourly RSI Change"))}
   Hourly Rising: {clean(row.get("Hourly RSI Rising"))}

🕒 Completed 15m:
   {clean(row.get("15m Completed Candle"))}

📊 15m RSI: <b>{fmt(row.get("15m RSI"))}</b>
   Previous: {fmt(row.get("15m Previous RSI"))}
   Change: {delta(row.get("15m RSI Change"))}
   Rising: {clean(row.get("15m RSI Rising"))}

🔄 15m Reversal:
   <b>{clean(row.get("15m RSI Reversal"))}</b>

🎯 Category: <b>{clean(category)}</b>
📢 Signal: {clean(row.get("Signal"))}

💡 {clean(row.get("Reason"))}
""".strip()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("latest_results.json not found.")

    data = load_json(DATA_FILE, [])

    if not isinstance(data, list):
        raise RuntimeError("latest_results.json must contain a JSON array.")

    state = load_json(STATE_FILE, {})

    sent = 0
    current_state = {}

    # First run: initialize the baseline without sending a flood
    # of WATCH alerts for stocks that were already in WATCH.
    initializing = not state

    for row in data:
        symbol = str(row.get("Symbol") or "").strip()
        category = str(row.get("Category") or "").upper().strip()

        if not symbol:
            continue

        current_state[symbol] = category

        previous = str(state.get(symbol, "")).upper()

        # ------------------------------------------------------
        # NEW SETUP
        # ------------------------------------------------------
        if category == "SETUP" and previous != "SETUP":
            telegram_send(build_message(row, "NEW_SETUP"))
            sent += 1
            print(f"Telegram: NEW SETUP -> {symbol}")

        # ------------------------------------------------------
        # NEW WATCH
        # ------------------------------------------------------
        elif (
            not initializing
            and category == "WATCH"
            and previous not in ("WATCH", "SETUP")
        ):
            telegram_send(build_message(row, "NEW_WATCH"))
            sent += 1
            print(f"Telegram: NEW WATCH -> {symbol}")

        # ------------------------------------------------------
        # SETUP LOST
        # ------------------------------------------------------
        elif (
            not initializing
            and previous == "SETUP"
            and category != "SETUP"
        ):
            # Do not send a WATCH notification here because
            # a SETUP -> WATCH transition is already represented
            # by this event.
            telegram_send(build_message(row, "SETUP_LOST"))
            sent += 1
            print(f"Telegram: SETUP LOST -> {symbol}")

    save_json(STATE_FILE, current_state)

    print(f"Telegram alerts sent: {sent}")
    print(f"Alert state saved: {STATE_FILE}")


if __name__ == "__main__":
    main()
