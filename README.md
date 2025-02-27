# Fairness in Machine Learning - Research Project

## 📌 Project Overview
This project aims to reduce bias in machine learning models by implementing fairness-aware methodologies at different stages of the machine learning pipeline. The solution integrates multiple fairness-enhancing techniques at the **preprocessing, in-processing, and postprocessing** levels to balance model performance and fairness.

## 📂 Repository Structure
```
Final_DS_Project/
│── all_results.txt                 # All experiment results
│── fairness_functions.py           # Fairness-related utility functions
│── pipeline_adult.ipynb            # Fairness evaluation on the Adult dataset
│── pipeline_bank.ipynb             # Fairness evaluation on the Bank dataset
│── /pipeline_compas(example_notebook).ipynb  # Fairness evaluation on the COMPAS dataset
│── pipeline_german_credit.ipynb    # Fairness evaluation on the German Credit dataset
│── requirements.txt                # Project dependencies
│── Final_Report.pdf                # Research report (Newly added)
```

## 🚀 How to Run the Project
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/car_insurance_prediction.git
cd car_insurance_prediction/Final_DS_Project
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebooks
You can open and execute any of the notebooks inside `Final_DS_Project/`:
```bash
jupyter notebook
```

## 📊 Results & Findings
The results of all models are stored in **all_results.txt**, including fairness metrics:
- **Demographic Parity Difference**
- **Equalized Odds Difference**
- **Accuracy & F1-score**

For a detailed analysis, please refer to **Final_Report.pdf**.

## 📄 Citation & References
This project builds upon fairness research from:
- Verma & Rubin (2018) - "Fairness Definitions Explained"
- Hardt et al. (2016) - "Equality of Opportunity in Supervised Learning"
- Fairlearn library documentation

## 👥 Contributors
- **Danielle Levy**
- **Jonathan Mandl**




