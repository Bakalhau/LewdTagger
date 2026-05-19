# AniTagger

AniTagger is a high-performance, local web application for automatic anime image tagging and character identification. The tool uses the **WD 1.4 SwinV2 Tagger V2** artificial intelligence model, running via ONNX Runtime for blazing-fast backend processing.

## Key Features
- **Recursive Scanning:** Search for thousands of images across folders and subfolders.
- **Automatic Tagging:** Detects *General Tags*, *Characters*, and *Ratings* (Safe/Questionable/Explicit).
- **F95 Tags Matching:** Automatically compares detected tags against a database of 64 F95zone categories, visually displaying the source images for each matched classification.
- **Batch Export:** Export all detected tags in bulk to a `.zip` file containing individual `.txt` documents.

---

## How to Install and Run

### Prerequisites
- **Python 3.10+** installed on your system.

### Step 1: Install dependencies
Open a terminal in the project folder and run the following command to install FastAPI, ONNX Runtime, Uvicorn, Pillow, and other requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Run the server
Still in the project terminal, start the backend:
```bash
python main.py
```
> *On the first run, the system will automatically download the AI model from HuggingFace (~100MB).*

### Step 3: Access the interface
Open your preferred web browser and go to:
[http://localhost:8000](http://localhost:8000)

---

## ⚡ Performance: CPU vs GPU

By default, AniTagger runs using your **CPU**. While this works perfectly fine and requires no special setup, you can unlock significantly faster tagging speeds by using a dedicated **GPU** (like an NVIDIA RTX card).

To enable GPU acceleration, simply uninstall the default ONNX package and install the GPU version:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
The application will automatically detect your GPU and enable hardware acceleration, drastically reducing the time it takes to process large batches of images.
