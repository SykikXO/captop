<div align="center">
  <img src="assets/vtopautocaptcha.png" alt="VTOP AutoCaptcha" width="250" />
  <h1>captop / VTOP AutoCaptcha</h1>
</div>

**Captop** is a complete, open-source pipeline for researchers and ML enthusiasts to understand how to collect, label, and train models on real-world captcha data.

## Installation Guide

The model is available as a browser extension to auto-solve captchas directly on the VTOP login page!

### Browser & Platform Compatibility

| Platform | Browser | Install Method | Status |
|:---------|:--------|:---------------|:-------|
| 🖥️ **Windows / macOS / Linux** | Firefox | [Add-ons Store](https://addons.mozilla.org/en-US/firefox/addon/vtop-captcha-bye-bye/) | ✅ One-click install |
| 🖥️ **Windows / macOS / Linux** | Chrome | `.zip` → Load unpacked | ✅ Works |
| 🖥️ **Windows / macOS / Linux** | Edge | `.zip` → Load unpacked | ✅ Works |
| 🖥️ **Windows / macOS / Linux** | Brave | `.zip` → Load unpacked | ✅ Works |
| 🖥️ **Windows / macOS / Linux** | Vivaldi | `.zip` → Load unpacked | ✅ Works |
| 🖥️ **Windows / macOS / Linux** | Opera | `.zip` → Load unpacked | ✅ Works |
| 📱 **Android** | Firefox | [Add-ons Store](https://addons.mozilla.org/en-US/firefox/addon/vtop-captcha-bye-bye/) | ✅ One-click install |
| 📱 **Android** | Kiwi Browser | `.crx` direct install | ⚠️ Archived (no updates) |
| 📱 **Android** | Yandex Browser | `.crx` direct install | ✅ Works |
| 📱 **Android** | Lemur Browser | `.crx` direct install | ✅ Works |
| 📱 **Android** | Edge Canary | `.crx` manual install | ✅ Works |

---

### 🦊 Firefox (Desktop & Android)

Install directly from the official Add-ons store — works on both desktop and mobile Firefox:

> **[⬇️ Download: VTOP Captcha Bye Bye](https://addons.mozilla.org/en-US/firefox/addon/vtop-captcha-bye-bye/)**

---

### 💻 Chromium-based Browsers (Desktop)

Works on **Chrome, Edge, Brave, Vivaldi, Opera**, and any Chromium-based browser:

1. Go to the [**Releases**](https://github.com/SykikXO/captop/releases) page and download `captop-chrome-vX.X.X.zip`
2. **Extract** the `.zip` into a folder
3. Open your browser and navigate to `chrome://extensions` (or `edge://extensions`, `brave://extensions`, etc.)
4. Toggle **Developer mode** ON (top-right corner)
5. Click **"Load unpacked"** and select the extracted folder
6. Done! Navigate to VTOP login and the extension will auto-solve captchas

---

### 📱 Android (Chromium-based)

Several Android browsers support Chrome extensions via `.crx` files:

**Yandex Browser / Lemur Browser:**
1. Download `captop-chrome.crx` from the [**Releases**](https://github.com/SykikXO/captop/releases) page
2. Open the downloaded `.crx` file — the browser will prompt you to install it
3. Confirm the installation and you're good to go!

**Kiwi Browser** *(archived — no longer receiving updates):*
1. Download `captop-chrome.crx` from [**Releases**](https://github.com/SykikXO/captop/releases)
2. Go to `chrome://extensions` in Kiwi
3. Enable **Developer mode** and tap **"+(from .crx/.zip/.user.js)"**
4. Select the downloaded `.crx` file

**Edge Canary (Android):**
1. Download `captop-chrome.crx` from [**Releases**](https://github.com/SykikXO/captop/releases)
2. Go to `edge://extensions` and enable **Developer mode**
3. Install the `.crx` file manually

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
