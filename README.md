
# 📊 Walmart Store Sales Forecasting

This project forecasts weekly sales for Walmart stores using historical sales data, economic factors, and holiday information. It leverages data preprocessing, feature engineering, and machine learning (XGBoost) to make accurate predictions. The app is deployed via Streamlit for an interactive experience.

## 🔍 Problem Statement
Retailers like Walmart need accurate weekly sales forecasts to manage inventory, staffing, and logistics. This project predicts department-level weekly sales for each store using features like markdown events, holidays, temperature, fuel prices, CPI, and unemployment rate.

## 📦 Dataset
The dataset comes from the [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) competition on Kaggle and includes:

- `train.csv`: Historical sales data
- `test.csv`: Test data for submission
- `features.csv`: Additional features like temperature, CPI, etc.
- `stores.csv`: Store types and sizes
- `sampleSubmission.csv`: Format for final predictions

## 🧠 Approach

- Merged datasets on Store, Date, and Dept
- Performed preprocessing: date parsing, missing value imputation, and encoding
- Engineered features like holiday effect and store type
- Trained an XGBoost Regressor for sales prediction
- Deployed a Streamlit app for live input and prediction

## 🚀 Streamlit App

🔗 Live App: [Click to launch the app](https://demand-forecasting-frqtr9fvefnqzxoyjdweni.streamlit.app/)

### How to Use
1. Select store, department, and other input features.
2. Click "Predict Weekly Sales" to see the forecast.

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-Learn, XGBoost)
- Streamlit (Web App Interface)
- GitHub + Streamlit Cloud (Deployment)

## 📁 Project Structure
```
├── app.py
├── model/
│   ├── xgb_model.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── requirements.txt
└── README.md
```

## 🖥️ Run Locally

```bash
git clone https://github.com/Yashv13/demand-forecasting.git
cd demand-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## ✍️ Author
Yash Vora  
[GitHub](https://github.com/Yashv13)
