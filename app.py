import nltk
nltk.download('punkt')

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load the model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize NLTK tools
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    from nltk.tokenize import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in tokens if word not in stop_words and word.isalpha()
    ]
    return " ".join(cleaned_tokens)

# Streamlit UI
st.title("Fake News Detection")

# Input text box
news_text = st.text_area("Enter News Article:")

# Prediction button
if st.button("Predict"):
    if news_text:
        # Preprocess and predict
        clean_text = preprocess_text(news_text)
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)
        
        if prediction[0] == 1:
            st.write("**Prediction**: Real News")
        else:
            st.write("**Prediction**: Fake News")
    else:
        st.write("Please enter some text to predict.")

model = joblib.load("fake_news_model.pkl")

