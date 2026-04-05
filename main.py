import pandas as pd 
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

st.title("Fake Job Detection System!!")

@st.cache_resource
def load_model():
    df = pd.read_csv("fake_job_postings_small.csv").fillna("")
    df['text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements']

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        return text
    
    df['text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['fraudulent']
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)

    # model = LogisticRegression()
    # model.fit(X, y)

    return model, vectorizer, clean_text

model, vectorizer, clean_text = load_model()

job_text = st.text_area("Paste the job description here!!")

if st.button("Check Job"):
    if job_text.strip():
        text = clean_text(job_text)
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("The job Posting is looks like 'Fake' ")

        else:
            st.success("The job posting is looks like 'Real' ")

    else:
        st.warning("Please paste the job description")
 