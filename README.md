# Captop

Thought of a small project to learn ML stuff, so I built a crowd-sourced captcha labeling system for building training datasets. 

**The dataset will be made public as soon as labeling reaches 100 percent (811/811). Watch/contribute at [Cloudflare Worker (VITBPL bypass)](https://captop.sykik.workers.dev) or [PythonAnywhere (NO VITBPL bypass)](https://sykik.pythonanywhere.com).**

## Project Structure

- **data/**: Contains captcha images and test data.
- **server/**: Flask backend and web interface for crowdsourcing labels for the dataset.
- **scripts/**: Utility scripts for database management, ML training, and relabeling.
- **worker/**: Cloudflare Worker proxy configuration.

## Setup

1. Install dependencies (Flask, etc).
2. Run database initialization: `python3 scripts/init_db.py`.
3. Start the server: `python3 server/app.py`.
