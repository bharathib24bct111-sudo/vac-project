🚀 Intrusion / Spam Detection Using Machine Learning (TF-IDF + NB / Logistic Regression)

This project builds a machine learning classifier to detect malicious or unwanted entries from a mixed-type dataset.
The pipeline automatically adapts to CSV files containing text, numerical, or mixed features, performs preprocessing, trains a model, and evaluates performance using multiple metrics and visualizations.

📌 Project Overview

The goal of this project is to create a flexible ML pipeline that:

Loads any uploaded dataset (CSV)

Automatically identifies text columns and numeric columns

Applies:

TF-IDF vectorization for text features

Standard scaling for numeric features

Trains a suitable model:

Complement Naive Bayes for text-only datasets

Logistic Regression for mixed-feature datasets

Produces detailed evaluation metrics:

Accuracy

Confusion matrix

Classification report

ROC Curve (Binary or Multiclass)

The entire workflow is implemented inside Google Colab so anyone can run it easily.

🧩 Dataset Description

The script accepts any dataset containing a label column named:

Label, label, target, class, or Class


The rest of the columns are treated as features.

✔ Supported Feature Types
Feature Type	Handling Method
Text (string/object)	TF-IDF vectorization
Numeric	StandardScaler
Mixed	ColumnTransformer (TF-IDF + Scaling)
No text columns	Converts whole row into a text string
⚙️ Machine Learning Pipeline
1️⃣ Data Loading
uploaded = files.upload()
df = pd.read_csv(file_name)

2️⃣ Label Detection

Auto-detects the label column and separates features (X) and labels (y).

3️⃣ Automatic Feature Type Identification

Extracts text columns

Extracts numeric columns

If no text column exists → Concatenates entire row as text

4️⃣ Model Selection
Dataset Type	Model Used
Text-only	Complement Naive Bayes
Mixed (text + numeric)	Logistic Regression
5️⃣ Model Training

Pipeline built using:

TfidfVectorizer

StandardScaler

ColumnTransformer

Pipeline

6️⃣ Evaluation Metrics

Accuracy Score

Classification Report

Confusion Matrix (heatmap)

ROC Curve (binary)

Multiclass ROC (One-vs-Rest)

📊 Visualizations
✔ Confusion Matrix

Automatically plots a labeled heatmap showing predictions vs ground truth.

✔ Class Distribution

Shows the count of each class before training.

✔ ROC / AUC

Binary → Standard ROC curve

Multiclass → One-vs-Rest ROC curves

📁 Project Files
File	Purpose
YourNotebook.ipynb	Main ML training and evaluation notebook
Dataset.csv	Input dataset uploaded by the user
README.md	Documentation for GitHub
▶️ How to Run This Project
1. Open Google Colab

https://colab.research.google.com/

2. Upload the notebook

Run each cell one by one.

3. Upload your CSV

The code automatically detects and processes your dataset.

4. View Results

The notebook will output:

Accuracy

Confusion Matrix

ROC Curve

Predictions

🔍 Example Results (Sample)
Accuracy: 0.94

Precision, Recall, F1-score:
---------------------------------
Class A → 0.93 / 0.94 / 0.94
Class B → 0.95 / 0.94 / 0.94


Confusion matrix and ROC graphs are displayed within the notebook.

🛠 Future Improvements

Add SMOTE or class-weight balancing

Save and load trained models

Add feature importance for TF-IDF

API for real-time predictions

Deploy model using Flask/FastAPI

