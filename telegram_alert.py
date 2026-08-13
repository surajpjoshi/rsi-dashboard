import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("latest_results.json")

STATE_FILE = Path("telegram_alert_state.json")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# ============================================================
# TELEGRAM SEND
# ============================================================

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


    request = urllib.request.Request(

        url,

        data=payload,

        method="POST"

    )


    with urllib.request.urlopen(request, timeout=30) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )


    if not result.get("ok"):

        raise RuntimeError(
            f"Telegram API error: {result}"
        )


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path, default):

    if not path.exists():

        return default


    try:

        with path.open("r", encoding="utf-8") as f:

            return json.load(f)


    except Exception:

        return default


# ============================================================

def save_json(path, data):

    with path.open("w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# HTML SAFETY
# ============================================================

def clean(value):

    if value is None:

        return "--"


    text = str(value)


    return (

        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")

    )


# ============================================================
# NUMBER HELPERS
# ============================================================

def num(value):

    try:

        return float(value)

    except (TypeError, ValueError):

        return None


# ============================================================

def fmt(value):

    n = num(value)


    if n is None:

        return clean(value)


    return f"{n:.2f}"


# ============================================================

def delta(value):

    n = num(value)


    if n is None:

        return "--"


    return f"{n:+.2f}"


# ============================================================
# BUILD SETUP MESSAGE
# ============================================================

def build_setup_message(row):

    symbol = clean(
        row.get("Symbol")
    )


    return f"""
🔥 <b>NEW RSI SETUP</b>

━━━━━━━━━━━━━━━━━━━━

<b>{symbol}</b>

💰 LTP: ₹{fmt(row.get("Current LTP"))}

📅 <b>Weekly RSI</b>
   RSI: {fmt(row.get("Current Week RSI"))}
   Change: {delta(row.get("Weekly RSI Change"))}

⏱ <b>Hourly RSI</b>
   RSI: {fmt(row.get("Current Hourly RSI"))}
   Change: {delta(row.get("Hourly RSI Change"))}
   Rising: {clean(row.get("Hourly RSI Rising"))}

🎯 Category: <b>SETUP</b>

📢 Signal: 🔥 SETUP

💡 {clean(row.get("Reason"))}

━━━━━━━━━━━━━━━━━━━━

⚠️ RSI signal only — wait for price/confirmation.
""".strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("==========================================")

    print("TELEGRAM RSI SETUP ALERT")

    print("==========================================")

    print()


    # ========================================================
    # CHECK DATA FILE
    # ========================================================

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            "latest_results.json not found."
        )


    # ========================================================
    # LOAD SCANNER DATA
    # ========================================================

    data = load_json(
        DATA_FILE,
        []
    )


    if not isinstance(data, list):

        raise RuntimeError(
            "latest_results.json must contain a JSON array."
        )


    print(
        f"Scanner records found: {len(data)}"
    )

    print()


    # ========================================================
    # LOAD PREVIOUS TELEGRAM STATE
    # ========================================================

    state = load_json(
        STATE_FILE,
        {}
    )


    if not isinstance(state, dict):

        state = {}


    # ========================================================
    # TRACK CURRENT STATE
    # ========================================================

    current_state = {}


    alerts_sent = 0

    setup_count = 0


    # ========================================================
    # PROCESS EVERY STOCK
    # ========================================================

    for row in data:


        symbol = str(
            row.get("Symbol") or ""
        ).strip()


        category = str(
            row.get("Category") or ""
        ).upper().strip()


        # ----------------------------------------------------
        # Skip invalid rows
        # ----------------------------------------------------

        if not symbol:

            continue


        # ----------------------------------------------------
        # Save current state
        # ----------------------------------------------------

        current_state[symbol] = category


        # ----------------------------------------------------
        # Previous state
        # ----------------------------------------------------

        previous = str(
            state.get(symbol, "")
        ).upper().strip()


        # ====================================================
        # ONLY SETUP MATTERS
        # ====================================================

        if category != "SETUP":

            print(
                f"{symbol}: {category} → NO TELEGRAM ALERT"
            )

            continue


        setup_count += 1


        # ====================================================
        # NEW SETUP
        #
        # Send only when previous state was NOT SETUP.
        #
        # Examples:
        #
        # WATCH  → SETUP  = SEND
        # WAIT   → SETUP  = SEND
        # IGNORE → SETUP  = SEND
        # blank   → SETUP  = SEND
        #
        # SETUP  → SETUP  = DON'T SEND
        # ====================================================

        if previous != "SETUP":

            print(
                f"{symbol}: {previous or 'NONE'} → SETUP"
            )

            print(
                "  🔥 Sending Telegram alert..."
            )


            message = build_setup_message(row)


            telegram_send(message)


            alerts_sent += 1


            print(
                f"  ✅ Telegram alert sent: {symbol}"
            )


        else:

            print(
                f"{symbol}: SETUP → SETUP"
            )

            print(
                "  ⏭ Already alerted. Skipping."
            )


    # ========================================================
    # SAVE CURRENT STATE
    # ========================================================

    save_json(
        STATE_FILE,
        current_state
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("==========================================")

    print("TELEGRAM ALERT SUMMARY")

    print("==========================================")

    print(
        f"Scanner records : {len(data)}"
    )

    print(
        f"SETUP stocks    : {setup_count}"
    )

    print(
        f"Alerts sent     : {alerts_sent}"
    )

    print(
        f"State file      : {STATE_FILE}"
    )

    print("==========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
