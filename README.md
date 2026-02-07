# Captop

A crowd-sourced captcha labeling system for building ML training datasets.

## Project Status: 🏛️ Crowdsourcing Complete (Feb 5, 2026)

**The crowdsourcing phase to label the captcha dataset is complete!** ML models trained on this data will be uploaded soon.

📊 **[View Full Analytics Report](server/log_analysis.md)** — Traffic patterns, geographic distribution, and system metrics.

## Project Structure

- **data/**: Labeled captcha datasets (Available as 200, 500, and 811 image zips)
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

The individual images have been organized into zip files for convenience:
- `data/captchas/dataset_200.zip`: 200 random labeled images.
- `data/captchas/dataset_500.zip`: 500 random labeled images.
- `data/captchas/dataset_811.zip`: The full dataset of 811 labeled images.
- `data/dataset_test.zip`: The dataset specifically for testing.

Individual labeled images are maintained locally for extraction and testing but are not tracked in the repository to keep it clean.

## Analytics

- `server/log_analysis.md` — Full traffic and contributor analytics
- `server/database_summary.md` — Database statistics
- `server/india_map.svg` — Geographic traffic visualization
