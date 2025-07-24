<p align="center">
  <img src="icon.jpg" alt="Sallot27 Logo" width="150"/>
</p>

# 🧠 Kaggle Competition Solutions

This repository contains my end-to-end solutions for various Kaggle competitions. Each folder contains the full pipeline — including data analysis, preprocessing, model training, evaluation, and final submission generation.

---

## 🚀 Competitions Covered

### 1. 🛳️ Titanic: Machine Learning from Disaster
- **Link**: [Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **Goal**: Predict which passengers survived the Titanic shipwreck.
- **Techniques**:
  - Logistic Regression, Random Forest, XGBoost
  - Feature engineering: titles, family size
  - Submission ready `.csv` file
- **Notebook**: [`titanic-notebook.ipynb`](./titanic/titanic-notebook.ipynb)

---

### 2. 🏠 House Prices: Advanced Regression Techniques
- **Link**: [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Goal**: Predict final sale prices of homes in Ames, Iowa.
- **Techniques**:
  - XGBoost Regressor
  - Log-transform of target variable
  - Missing value imputation and categorical encoding
  - Submission CSV creation for Kaggle
- **Notebook**: [`house-prices.ipynb`](./house-prices/house-prices.ipynb)

---

### 3. 🪐 Spaceship Titanic
- **Link**: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- **Goal**: Predict whether passengers were transported to another dimension.
- **Techniques**:
  - Ensemble model: XGBoost + LightGBM
  - Feature engineering: cabin deck/side, spend totals
  - Validation accuracy ~0.798
  - Final `.csv` file for submission
- **Notebook**: [`spaceship-titanic.ipynb`](./spaceship-titanic/spaceship-titanic.ipynb)

---

### 4. 🧠 LLM Prompt Classification - Fine-Tuning
- **Link**: [LLM Prompt Classification](https://www.kaggle.com/competitions/llm-classification-finetuning)
- **Goal**: Classify prompts into correct categories to aid large language model fine-tuning.
- **Techniques**:
  - NLP text preprocessing
  - BERT-based transformer fine-tuning (e.g., `roberta-base`)
  - Tokenization, truncation, padding
  - Multi-class classification with Softmax
- **Tools**: PyTorch, HuggingFace Transformers
- **Notebook**: [`llm-prompt-classification.ipynb`](./llm-classification/llm-prompt-classification.ipynb)

---
## 📁 Repository Structure

.
├── titanic/
│ └── titanic-notebook.ipynb
│ └── submission.csv
│
├── house-prices/
│ └── house-prices.ipynb
│ └── submission.csv
│
├── spaceship-titanic/
│ └── spaceship-titanic.ipynb
│ └── submission.csv
│
├── llm-classification/
│   └── llm-prompt-classification.ipynb
│   └── submission.csv
|
└── README.md



---

## 🛠️ Requirements

To run the notebooks, install:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
Use Jupyter Notebook or any Python IDE to run and explore the notebooks.

📫 Contact
Feel free to reach out via Kaggle Profile or GitHub Issues if you have questions or suggestions!
