# Animals Detection in Backyard

Real-time backyard wildlife monitoring using **TensorFlow SSD MobileNetV2**, **OpenCV**, and optional **Twilio SMS** alerts.  

## Features
- Live video inference (webcam, file, or RTSP stream).
- Detects common animals (cat, dog, bird, horse, cow, sheep, elephant, bear, zebra, giraffe) and **person**.
- Configurable confidence threshold.
- Optional Twilio SMS alerts with per-class cooldown.

## Quick Start
```bash
# 1) Clone and enter
git clone https://github.com/<your-username>/animals-in-backyard.git
cd animals-in-backyard

# 2) (optional but recommended) create virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Configure
cp .env.example .env
# Edit .env to set video source, thresholds, and Twilio (optional)

# 5) Run
python app.py
