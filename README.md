# 🌧 Rainfall Prediction ML Project

## 📌 Overview

This project predicts whether rainfall will occur based on weather parameters using multiple machine learning models.
It compares different algorithms and provides predictions through a simple frontend interface.

The system evaluates and deploys the following models:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* XGBoost

---

## 🎯 Objective

To build a machine learning system that:

* predicts rainfall (Yes/No)
* compares multiple models
* evaluates performance metrics
* visualizes confusion matrices
* allows manual input prediction through UI

---

## 📊 Features Used

The model is trained on these weather parameters:

* Pressure
* Temperature
* Dewpoint
* Humidity
* Cloud
* Sunshine
* Wind Direction
* Wind Speed

Target variable:

```
Rainfall → Yes / No
```

---

## 🧠 Models Implemented

| Model               | Purpose                       |
| ------------------- | ----------------------------- |
| Logistic Regression | Baseline classifier           |
| KNN                 | Distance-based classification |
| Decision Tree       | Rule-based classification     |
| Random Forest       | Ensemble tree model           |
| XGBoost             | Boosted tree model            |

---

## 📈 Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🖥 Interface

The project includes a frontend built using **Streamlit** where users can:

* enter custom weather values
* run prediction
* view outputs from all models

---

## 🚀 How to Run Project

### 1️⃣ Clone repository

```
git clone https://github.com/abinashbir/Railfall.git
cd Project_rainfall
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 🔮 Example Prediction

Input:

```
Pressure = 1018
Temperature = 25
Humidity = 85
...
```

Output:

```
RandomForest → Yes
Logistic → No
KNN → Yes
```

---

## 📂 Project Structure

```
Project_rainfall/
│
├── app.py
├── predict.py
├── all_models.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

---

## 🧪 Model Training Pipeline

1. Data preprocessing
2. Feature scaling (for selected models)
3. Train-test split
4. Model training
5. Evaluation
6. Saving best model

---

## 📌 Key Learning Outcomes

This project demonstrates:

* end-to-end ML pipeline
* model comparison
* evaluation metrics
* deployment basics
* frontend integration

---

## 🏆 Best Performing Model

After testing multiple algorithms, the best performing model was:

```
Random Forest Classifier
```

because it achieved the highest test accuracy and lowest false predictions.

---

## 📚 Libraries Used

```
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
joblib
streamlit
fastapi
uvicorn
```

---

## 👨‍💻 Author

**Abinash Bir**

---

## 📜 License

This project is for academic and educational use.

---

## ⭐ Future Improvements

* Add live weather API integration
* Deploy model online
* Add probability visualization
* Add model selection option in UI

---

**If you like this project, consider giving it a ⭐**
