Fake Job Posting Detection System
Overview
This project detects fraudulent job postings using Machine Learning and Natural Language Processing (NLP). Users can paste any job description and the system will predict whether it is a Real or Fake job posting.
Live Demo
🔗 Click here to try the app
Technologies Used

Python
Pandas, NumPy
NLP (TF-IDF Vectorizer)
Scikit-learn (Logistic Regression)
Streamlit

How It Works

Dataset is loaded and cleaned using NLP techniques
TF-IDF Vectorizer converts text into numerical features
Logistic Regression model is trained with class_weight='balanced' to handle imbalanced data
User inputs a job description and the model predicts Real or Fake

Dataset
Fake Job Postings Dataset from Kaggle containing real and fraudulent job listings
Model Performance

Algorithm: Logistic Regression
Accuracy: ~90%
Handles class imbalance using class_weight='balanced'

How to Run Locally
bashpip install pandas scikit-learn streamlit
streamlit run app.py
Project Structure
Fake_Job_Posting_Detection/
├── app.py
├── fake_job_postings_small.csv
├── requirements.txt
└── README.md
About
Made by Yash | BCA Student
