# Captop

**Captop** is a complete, open-source pipeline for researchers and ML enthusiasts to understand how to collect, label, and train models on real-world captcha data.

## The Story

This project started as a personal journey to learn Machine Learning. I wanted to work on something "unexplored" and real.

1. **The Hunt**: I used some **JS-hackery** to scrape and collect a raw dataset of captchas directly from my college's website.
2. **The Crowdsource**: Since the data was unlabeled, I built a lightweight, full-stack application to crowdsource the labels. This allowed friends and contributors to help build the ground truth dataset.
3. **The Result**: After collecting over 800 labels and training a high-performance **CRNN (CNN+GRU)** model, I've reached the goal. The model now decodes these captchas with near 100% accuracy.

Now that the mission is complete, I've made the entire stack—from the scraping logic to the final trained model—**fully open-source**.

---

## Project Structure

- **data/**: Labeled captcha datasets (Available as 200, 500, and 811 image zips).
- **models/**: The final trained weights (`.pth`), performance charts, and quantization scripts.
- **scripts/**: The core logic for training, decoding, and data utility.
- **server/**: The Flask-based crowdsourcing platform and analytics dashboard.
- **worker/**: Cloudflare Worker proxy configuration.

## Dataset

Access the labeled data for your own projects:
- `data/captchas/dataset_811.zip`: The full labeled dataset (811 images).
- `data/dataset_test.zip`: Unlabeled images used for final model verification.

## Performance & Usage

The model achieves a **Validation Loss: 0.0013**. 
- See **[models/README.md](models/README.md)** for loss charts and benchmarks.
- See **[MODEL_USAGE.md](MODEL_USAGE.md)** for pseudo-code on how to integrate the model into your own scripts.

## Analytics

Detailed insights from the crowdsourcing phase:
- **[View Analytics Report](server/log_analysis.md)** — Contributor stats, traffic maps, and system performance.
- `server/india_map.svg` — Geographic distribution of our contributors.
