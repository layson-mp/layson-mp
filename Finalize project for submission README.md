## 📉 Customer Churn Prediction Project – End-to-End ML & Streamlit App

### 🔍 Overview

This project builds an end-to-end machine learning solution to predict customer churn and deploys the best-performing model as an interactive **Streamlit web application**. The system helps stakeholders identify high-risk customers early and run **what-if simulations** to support retention strategies.

---

## 🚀 Features

* 🔮 Real-time **churn probability prediction**
* 📊 **Probability visualization**
* 📌 **Feature importance dashboard**
* 🔁 **What-if scenario simulator**
* 📈 Model evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC
* 🧠 Model explainability

---

## 🗂️ Project Structure

```text
.
├── app.py                  # Streamlit web app
├── churn_model.pkl         # Trained ML model (joblib)
├── feature_names.pkl       # Feature schema used by the model
├── churn_project.ipynb     # Google Colab / Jupyter notebook
├── data/                   # (optional) dataset
├── README.md               # Project documentation
```

---

## 🧪 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone the repo

```bash
git clone https://github.com/layson-mp/Customer-Churn-Prediction-Project.git
cd Customer-Churn-Prediction-Project
```

### 2️⃣ Install dependencies

```bash
python -m pip install -r requirements.txt
```

Or manually:

```bash
python -m pip install streamlit pandas scikit-learn matplotlib joblib
```

### 3️⃣ Run the Streamlit app

```bash
python -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 Model & Methodology

* Data preprocessing (missing values, encoding, scaling)
* Feature engineering:

  * Balance-to-Salary ratio
  * Product density indicator
  * Engagement-product interaction
  * Age-tenure interaction
* Models evaluated:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
* Best model selected using **ROC-AUC** and **F1-score**

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

## 🖥️ Streamlit Dashboard

The Streamlit app allows users to:

* Input customer features
* Adjust engagement and product usage
* Observe churn probability changes in real time
* Explore feature importance

---

## 📌 Business Impact

* Early detection of customers at risk of churn
* Supports **data-driven decision making**
* Helps design targeted retention strategies
* Improves customer lifetime value

---

## 🛣️ Future Improvements

* SHAP explanations
* CSV batch prediction upload
* Cloud deployment (Streamlit Cloud / HuggingFace Spaces)
* Model retraining pipeline
* Role-based dashboards for stakeholders

---

## 👤 Author

**Layson Mpalanga**
GitHub: [https://github.com/layson-mp](https://github.com/layson-mp)

---

## 📄 License

This project is for academic and educational purposes.

---

### 📦 Optional: `requirements.txt`

Create a file called `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
```
