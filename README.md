# NADE-beyond-pictures

This repository explores a Neural Autoregressive Distribution Estimator (NADE) applied to data beyond images.

- Part_A_text.ipynb: a compact text experiment that turns a small vocabulary into binary indicators and trains a NADE-style model.
- Part_B_speach.ipynb: a speech-based version that converts short audio clips into spectrogram-like binary features.

The notebooks are designed as beginner-friendly experiments for understanding autoregressive modeling on non-image data.

## Setup (Windows)

From the repository folder, create and activate a virtual environment, then install the notebook dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name nade-beyond-pictures --display-name "Python (NADE beyond pictures)"
```

In VS Code, select the `.venv` Python interpreter and the **Python (NADE beyond pictures)** kernel for the notebooks. PyTorch automatically uses CUDA when a compatible CUDA-enabled installation and GPU are available; otherwise, the notebooks run on CPU.
