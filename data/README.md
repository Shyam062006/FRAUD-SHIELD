# PaySim Synthetic Financial Dataset

This directory is designated for the PaySim mobile money dataset used for training and benchmarking the FRAUD SHIELD machine learning pipeline.

> **Note on Repository Hygiene:**  
> The raw and cleaned dataset files (`dataset.csv` ~470 MB, `clean_dataset.csv` ~150 MB) exceed standard Git tracking limits and are omitted from the repository in accordance with version control best practices. All dataset files are ignored via `.gitignore`.
> 
> The serialized model (`ML/model.pkl`, 740 KB) and scaler (`ML/scaler.pkl`, 1.2 KB) are bundled directly with the repository, enabling full out-of-the-box local execution and API inference without requiring data downloads.

---

## Dataset Origin

- **Source:** [Kaggle - Synthetic Financial Datasets For Fraud Detection (PaySim)](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Paper Citation:** E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, *"PaySim: A financial mobile money simulator for fraud detection"*, in The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus, 2016.
- **License:** CC BY-SA 4.0

---

## Download Instructions (To Retrain from Scratch)

If you wish to re-execute the training notebooks ([`ML/experimentation_and_benchmark.ipynb`](../ML/experimentation_and_benchmark.ipynb) and [`ML/train_final_model.ipynb`](../ML/train_final_model.ipynb)):

1. **Download the Dataset:**
   - Navigate to Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
   - Click **Download** (requires a free Kaggle account) to obtain `PS_20174392719_1491204439457_log.csv` (~470 MB).

2. **Place in Project:**
   - Rename or symlink the extracted CSV to `dataset.csv` inside this `data/` directory or project root:
     ```bash
     mv ~/Downloads/PS_20174392719_1491204439457_log.csv data/dataset.csv
     ```

3. **Run Training Notebooks:**
   - Execute [`ML/experimentation_and_benchmark.ipynb`](../ML/experimentation_and_benchmark.ipynb) to reproduce data filtering, train/test split, and multi-model benchmark evaluation.
   - Execute [`ML/train_final_model.ipynb`](../ML/train_final_model.ipynb) to retrain the champion Random Forest model and export fresh `model.pkl` and `scaler.pkl` artifacts.
