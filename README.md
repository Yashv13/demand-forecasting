
# 📈 Walmart Store Sales Forecasting

A machine learning application that predicts weekly sales for Walmart stores using historical sales and markdown data. Built with XGBoost and deployed using Streamlit.

🔗 **Live Demo**: [Streamlit App](https://demand-forecasting-frqtr9fvefnqzxoyjdweni.streamlit.app/)

---

## 🚀 Project Overview

Retail forecasting is essential for inventory planning, marketing strategies, and logistics. This project tackles a real-world Kaggle challenge hosted by Walmart, focusing on predicting weekly sales at a department level across 45 stores using historical data and markdown events.

---

## 📂 Project Structure

```
├── app.py                    # Streamlit application
├── model/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   ├── encoder.pkl           # OneHotEncoder for categorical features
│   ├── scaler.pkl            # StandardScaler for input features
│   └── feature_columns.pkl   # List of features used for training
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🧠 Model & Features

- **Model Used**: XGBoost Regressor
- **Feature Engineering**:
  - One-hot encoding of categorical variables (`Type`, `IsHoliday`)
  - Scaling of numerical features
  - Merging datasets (`train.csv`, `features.csv`, `stores.csv`)
  - Handling of missing values and date formatting

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Yashv13/demand-forecasting.git
cd demand-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Input Fields (App)

- `Store`: Integer (1–45)
- `Dept`: Department number
- `Temperature`: Float (°F)
- `Fuel_Price`: Float ($)
- `CPI`: Consumer Price Index
- `Unemployment`: %
- `IsHoliday`: Boolean (True/False)
- `Type`: Store type (`A`, `B`, or `C`)

---

## 📦 Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

(Full list in `requirements.txt`)

---

## 🙋‍♂️ Author

**Yash Vora**  
GitHub: [@Yashv13](https://github.com/Yashv13)

---

## 📜 License

This project is licensed under the MIT License.

---
