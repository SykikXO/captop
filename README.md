# Captop

A crowd-sourced captcha labeling system for building ML training datasets.

## Project Status: ✅ Complete (Feb 5, 2026)

**811/811 images labeled** with 3,353 submissions from 45 unique contributors.

📊 **[View Full Analytics Report](server/log_analysis.md)** — Traffic patterns, geographic distribution, and system metrics.

## Project Structure

- **data/**: Labeled captcha dataset (811 images)
- **server/**: Flask backend, database, and analytics
- **scripts/**: Utility scripts for database, renaming, and geolocation
- **worker/**: Cloudflare Worker proxy configuration

## Quick Start

```bash
pip install flask
python3 scripts/init_db.py
python3 server/app.py
```

## Dataset

The labeled dataset is in `data/captchas/` with filenames matching their labels (e.g., `A3K7BX.jpg`).

## Analytics

- `server/log_analysis.md` — Full traffic and contributor analytics
- `server/database_summary.md` — Database statistics
- `server/india_map.svg` — Geographic traffic visualization
