# Fake_True_News_NLP

Fake News Detection Using Machine Learning
This project builds and evaluates machine learning models to distinguish between fake and real news articles using the Fake and Real News Dataset. It involves extensive data preprocessing, model benchmarking, and interpretability analysis, balancing performance with transparency.

🔍 Overview
The goal is to classify news articles as fake or real using both textual content and metadata. The pipeline includes:

Baseline Modeling: Logistic Regression, Ridge Classifier, and Naive Bayes

Enhanced Modeling: XGBoost with SHAP interpretability

Text Processing: TF-IDF with unigrams & bigrams

Structured Features: Title & text length, punctuation ratio, sentiment score, subject encoding

Evaluation: Cross-validation, domain-wise performance, confusion matrix, ROC-AUC curve, SHAP summary plots

📁 Dataset
The dataset includes ~44,000 labeled articles across three domains:

News

Politics

Other (e.g., Government, World)

Sourced from Kaggle: Fake and Real News Dataset

📊 Results
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	1.00	1.00	1.00	1.00
Ridge Classifier	1.00	1.00	1.00	1.00
Naive Bayes (text)	0.95	0.95	0.95	0.95
XGBoost	1.00	1.00	1.00	1.00

✅ Key Highlights
Domain-Wise Bias Check: Confirmed consistent predictions across all subjects

Feature Importance: Identified key n-grams and metadata features driving predictions

Explainability: SHAP plots provide feature-level transparency

Model Robustness: Validated through cross-validation and ROC-AUC analysis

⚙️ Dependencies
To install all required packages:

bash
Copy
Edit
pip install -r requirements.txt
Dependencies include:
scikit-learn, xgboost, matplotlib, seaborn, shap, pandas, numpy

🚀 Getting Started
To reproduce results:

Download the dataset from Kaggle

Place Fake.csv and True.csv in the root directory

Run the notebook:
FakeNews_Classification.ipynb

📌 Future Work
Incorporate transformer-based models (e.g., BERT) for deeper linguistic representation

Explore time-based trends and topic modeling

Deploy model via Flask or Streamlit for real-time inference

