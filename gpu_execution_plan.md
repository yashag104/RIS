# GPU Execution Plan: Running the RIS FL Codebase on Another PC

This document outlines the step-by-step plan for running the RIS Federated Learning codebase on a secondary PC equipped with a GPU. Because the code is already configured to automatically use the GPU if available (via `torch.cuda.is_available()` in `config.py`), the primary task is correctly setting up the environment.

## 1. Transfer the Codebase
First, transfer the files to the GPU PC. Do not transfer the virtual environment or generated data.

* **Via Git (Recommended):** If your code is on GitHub/GitLab, simply `git clone <repo_url>` on the new PC.
* **Via ZIP / USB:** Compress the `RIS/` folder, but **exclude** the following directories to save time and prevent cross-platform issues:
  * `.venv/`
  * `__pycache__/`
  * `.pytest_cache/`
  * `results/` (Optional, unless you want to keep old results)

## 2. Prerequisites on the New PC
Ensure the new PC has the following installed:
* **NVIDIA Drivers:** The latest drivers for your GPU.
* **Python 3.8+:** Verify by running `python --version` or `python3 --version`.

*(Note: You do not strictly need to install the full CUDA toolkit manually, as PyTorch installs its own precompiled CUDA binaries, but having the NVIDIA drivers is mandatory).*

## 3. Set Up the Virtual Environment
Open a terminal (or command prompt / PowerShell) on the new PC, navigate to the copied `RIS` directory, and create a clean environment:

**Windows:**
```cmd
cd path\to\RIS
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
cd /path/to/RIS
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install GPU-enabled PyTorch
Before installing the rest of the requirements, install the GPU version of PyTorch. The command depends on the operating system and CUDA version. Go to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to get the exact command. 

For **CUDA 11.8** or **CUDA 12.1** (common on Linux/Windows):
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*(Verify it works by running: `python -c "import torch; print(torch.cuda.is_available())"` — it should print `True`.)*

## 5. Install Remaining Dependencies
Install the rest of the packages required for the project:
```bash
pip install -r requirements.txt
```

## 6. Run the Code
The project is already designed to detect the GPU automatically. In `config.py` (line 43), it contains:
`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

You can simply execute your experiments:
```bash
# To run the full experiment suite:
python run_all_experiments.py

# Or to run the main simulation:
python main.py
```

Monitor your GPU usage while running by opening a separate terminal and running `nvidia-smi` (or using Task Manager on Windows).
