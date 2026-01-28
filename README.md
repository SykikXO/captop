# Captop

A crowd-sourced captcha labeling system for building training datasets.

## Project Structure

- **data/**: Contains captcha images and test data.
- **server/**: Flask backend and web interface.
- **scripts/**: Utility scripts for database management, ML training, and relabeling.
- **worker/**: Cloudflare Worker proxy configuration.

## Setup

1. Install dependencies (Flask, etc).
2. Run database initialization: `python3 scripts/init_db.py`.
3. Start the server: `python3 server/app.py`.
