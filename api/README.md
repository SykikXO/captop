# Model Inference API

This directory contains the high-performance Rust backend used to serve the CAPTCHA model in production.

## Overview

The API is built using Rust and utilizes the ONNX runtime to efficiently perform inference on the trained model. It is designed to be fully containerized and easily deployable via Docker.

### Key Components

- **`src/`**: The Rust source code defining the web server endpoints and ONNX inference logic.
- **`model/`**: Contains the exported `.onnx` weights (`captcha_model.onnx`) loaded by the application at runtime.
- **`api_proxy_worker.js`**: A Cloudflare worker script configured to proxy traffic and handle CORS, pointing custom domains (like DuckDNS) directly to the API server.
- **`Dockerfile` & `entrypoint.sh`**: Configurations to bundle the Rust application into a lightweight container.

## Deployment

This API is designed to run automatically on a remote server (such as a DigitalOcean Droplet). Updates to the main branch or version bumps can trigger GitHub Actions to build the Docker image and deploy it directly over SSH to the Droplet.
