# PneumoDetect

**PneumoDetect** is a Deep Learning system to **detect pneumonia from chest X-ray images** using CNNs.  
It classifies X-rays as **Normal** or **Pneumonia** to assist doctors in faster diagnosis.

--

## Features
- Detects pneumonia automatically from chest X-rays  
- Provides metrics: accuracy, precision, recall, F1-score  
- Scripts for training, evaluation, and prediction  

---

## Dataset
- **Source:** [Kaggle Chest X-ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- Folder structure: `train/`, `val/`, `test/` with `NORMAL` and `PNEUMONIA`  
> Dataset is **not included**; download separately.

---

## Installation
```bash
git clone https://github.com/pari-dudeja2005/PneumoDetect.git
cd PneumoDetect
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt




