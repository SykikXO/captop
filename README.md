<div align="center">
  <img src="assets/vtopautocaptcha.png" alt="VTOP AutoCaptcha Project Image" />
  <h1>captop / VTOP AutoCaptcha</h1>
</div>

**Captop** is a complete, open-source pipeline for researchers and ML enthusiasts to understand how to collect, label, and train models on real-world captcha data.

## Installation Guide

The model is available as a browser extension to auto-solve captchas directly on the login page!

### 🦊 Firefox (Desktop & Android)
Install directly from the official Firefox Add-ons store:
> **[Download for Firefox: VTOP Captcha Bye Bye](https://addons.mozilla.org/en-US/firefox/addon/vtop-captcha-bye-bye/)**

### 💻 Chrome, Edge, Brave (Desktop)
Since this extension isn't on the Chrome Web Store, you can manually install it:
1. Go to the **Releases** page and download `captop-chrome-v1.3.1.zip`.
2. Extract the `.zip` file into a folder on your computer.
3. Open your browser and go to `chrome://extensions` (or `edge://extensions`).
4. Enable **Developer mode** (usually a toggle in the top-right corner).
5. Click **"Load unpacked"** and select the extracted folder. Done!

### 📱 Kiwi, Yandex, Lemur (Android Chromium)
Many Android Chromium forks support direct `.crx` installations!
1. Go to the **Releases** page and download `captop-chrome.crx`.
2. Tap the downloaded file or open it via the browser's extension page to install directly.

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
- **crowdsource/**: The Flask-based crowdsourcing platform and analytics dashboard.
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
- **[View Analytics Report](crowdsource/log_analysis.md)** — Contributor stats, traffic maps, and system performance.
- `crowdsource/india_map.svg` — Geographic distribution of our contributors.

## Contributors

A huge thanks to all the amazing people who helped crowdsource the data labels. This project wouldn't have been possible without you!

- [Aayush Chanda](https://github.com/Aayush-Chanda)
- [Abhishek](https://github.com/Ab705h)
- Aman
- Amritanshu Sahu
- [anand kr yadav](https://github.com/anand9608)
- Ansh
- [Arya](https://github.com/aryag-31)
- Aryan Agrahari
- [Blactract](https://github.com/CAPTAIN-BLACTRACT)
- Brijesh
- [Davood](https://github.com/TheDavood-10)
- [ffcs-planner-vitb.vercel.app](https://ffcs-planner-vitb.vercel.app/)
- Hardik
- Harshit
- [Kanishk](https://github.com/kanishk300)
- [Manov](https://github.com/Manov)
- Mayank
- Parth Sararthi
- Prateek
- [Pratyush](https://github.com/Pratyush-10)
- Puss in Boots
- [Raunak](https://github.com/Raunak-24)
- [Rishabh Bansal](https://github.com/Rishabh-Bansal)
- S
- [Sairam S](https://github.com/Sairam-S)
- [Sarthak](https://github.com/sarthak-01)
- Shaurya
- [Shivam](https://github.com/shivam-01)
- [Shreyas](https://github.com/shreyas-01)
- [Shubham](https://github.com/shubham-01)
- [Siddhant](https://github.com/siddhant-01)
- [Subal](https://github.com/ajsubal555)
- Sumit
- Sunidhi Suman
- [Suraj](https://github.com/suraj-01)
- [Tertiary Ion](https://github.com/TertiaryCo)
- urn.ab
- VANSHIKA
- [Vidishaa](https://github.com/vidishaa27)
- Vijay Naveen Mishra
- [Virat Nigam](https://github.com/viratnigam18)
- [Yash Priyam](https://github.com/Raunak-24)
