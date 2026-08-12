# RSI Dashboard

GitHub Actions runs the RSI scanner Monday-Friday every 15 minutes during the NSE session window 09:15-16:15 IST.

## Required GitHub Secret

Repository → Settings → Secrets and variables → Actions → New repository secret:

- Name: `UPSTOX_ACCESS_TOKEN`

Never put the Upstox token in source code.

## GitHub Pages

Enable Pages from the `main` branch and `/ (root)`. The dashboard is `index.html`.
