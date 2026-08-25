# RIS Federated Learning

This is a Federated Learning system for Reconfigurable Intelligent Surface (RIS) phase optimization.
RIS tiles (each with 64 pixels/elements) collaboratively learn to predict optimal phase shifts using FL, enabling SNR improvements in mmWave communication.

## Quick Start

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main training script:
```bash
python main.py
```