# 🏈 NFL Player Value Engine

> Predicting what NFL players should earn based on performance metrics and identifying who is overpaid or underpaid.

Built by **Darren Barkins** | MS Business Analytics, Cal Poly | Former D1 Cornerback

---

## 🚀 Live Demo

| Tool | Link |
|---|---|
| 🌐 Streamlit Web App | [nfl-value-engine.streamlit.app](https://nfl-value-engine.streamlit.app) |
| 📊 Tableau Dashboard | [View on Tableau Public](https://public.tableau.com/app/profile/darren.barkins/vizzes) |
| 💻 GitHub Repo | [github.com/darrenbarkins/nfl-value-engine](https://github.com/darrenbarkins/nfl-value-engine) |

---

## 📌 Project Overview

NFL contracts are worth hundreds of millions of dollars, but are they actually based on performance?

This project builds an end-to-end machine learning pipeline that:
- Pulls 6 years of real NFL player stats and contract data (2018–2023)
- Engineers 15 performance features to predict a player's fair market value (AAV)
- Identifies the most overpaid and underpaid players in the league
- Deploys findings in an interactive web app and Tableau dashboard

---

## 🔍 Key Findings

- **Passing TDs and passing yards** drive 48% of salary prediction power
- **Justin Herbert** was flagged as the most overpaid player relative to predicted performance value
- **Brock Osweiler** was flagged as the most underpaid player relative to predicted performance value
- **Age vs AAV correlation:** 0.340 — moderate relationship
- **Fantasy points vs AAV correlation:** 0.578 — stronger predictor than age alone
- QBs earn nearly **3x more** than the next highest paid position

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Collection | Python, nfl_data_py |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Visualization | Seaborn, Plotly, Tableau Public |
| Web App | Streamlit |
| Version Control | GitHub |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | XGBoost Regression |
| R² Score | 0.567 |
| RMSE | $5.31M |
| Training Records | 1,880 |
| Test Records | 471 |
| Features | 15 |

---

## 📁 Project Structure

- **data/** — Cleaned master dataset (model_df.csv)
- **notebooks/** — Data collection, EDA, and ML model notebooks
- **models/** — Trained XGBoost model and label encoder
- **app/** — Streamlit web app (app.py)
- **requirements.txt** — Python dependencies

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/darrenbarkins/nfl-value-engine.git
cd nfl-value-engine

# Create conda environment
conda create -n nfl-value-engine python=3.11
conda activate nfl-value-engine

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app/app.py
```

---

## 💡 Why I Built This

As a 3-school D1 cornerback and MS Business Analytics candidate, I understand both sides of athlete valuation on the field and in the data. This project combines my athletic background with my analytics skills to answer a question every NFL front office is asking:

> *"Are we paying our players what they're actually worth?"*

---

## 📬 Connect

- **LinkedIn:** [linkedin.com/in/darrenbarkins](https://linkedin.com/in/darrenbarkins)
- **GitHub:** [github.com/darrenbarkins](https://github.com/darrenbarkins)
- **Email:** darrenjbarkins@gmail.com or dbarkins@calpoly.edu
