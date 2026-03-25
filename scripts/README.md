# Scripts

Utility scripts for various project tasks.

- **train.py**: PyTorch script for training the captcha recognition model.
- **decode_captchas.py**: Utility for decoding individual captchas using the trained model.
- **export_onnx.py**: Exports the PyTorch `.pth` model to `.onnx` for use in the Rust backend.
- **quantize_model.py**: Utility to quantize the PyTorch model for better performance.
- **init_db.py**: Initializes the crowdsourcing SQLite database and populates images.
- **relabel_ollama.py**: Attempted using AI models (Ollama) to perform automated relabeling.
- **verify_labels.py**: Identifies mismatches between predictions and ground-truth labels.
- **log_analytics.py** & **ip_geolocate.py**: Analyze and extract analytics from the crowdsource backend.
- **rename_captchas.py** & **reorganize_dataset.py**: Helper routines for standardizing the dataset.
- **package.py**: Packages the project into a zip file for deployment.


### Training Pipeline

If you want to train your own model from scratch:

1. **Train Model:** Run `train.py` to initiate the training pipeline.
2. **Decode/Test:** Run `decode_captchas.py` to evaluate the model on individual images.
3. **Verify:** Use `verify_labels.py` to identify mismatches between the model's predictions and your ground-truth labels using a browser interface. This is also extremely useful to relabel captchas very quickly using human intelligence.
4. **Deploy:** Finally, use `export_onnx.py` to export your PyTorch model into the `.onnx` format required by the Rust API.