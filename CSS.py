import streamlit as st
import pandas as pd
import re
import nltk
import math
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# Download the VADER lexicon (one-time only)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Define sensational keywords (same as before)
sensational_keywords = [...]

# Functions (same as before)
def count_sensational_words(text): [...]
def count_hashtags(text): [...]
def count_emojis(text): [...]
def sentiment_vader(text): [...]
def process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status):
    # Some processing logic
    processed_tweet = some_processing_logic(tweet)

    if processed_tweet is None:
        print("Warning: Tweet processing failed!")
    return processed_tweet


# Streamlit UI with custom HTML and CSS
st.markdown(
    """
    <style>
    .stForm {
        background-color: cyan;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        line-color: Black;
    }
    .stTextArea, .stNumberInput, .stDateInput, .stCheckbox {
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom title
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-family: Arial, sans-serif;'>
        Twitter Misinformation Detection
    </h1>
    """,
    unsafe_allow_html=True
)

# Create a form for user input
with st.form("twitter_form"):
    # Input fields (same as before)
    tweet_body = st.text_area("Tweet Body", placeholder="Enter the tweet text here...")
    likes = st.number_input("Likes", min_value=0, value=0)
    retweets = st.number_input("Retweets", min_value=0, value=0)
    comments = st.number_input("Comments", min_value=0, value=0)
    quotes = st.number_input("Quotes", min_value=0, value=0)
    views = st.number_input("Views", min_value=0, value=0)
    followers = st.number_input("Number of Followers", min_value=0, value=0)
    followings = st.number_input("Number of Followings", min_value=0, value=0)
    joining_date = st.date_input("Date of Joining")
    verified_status = st.checkbox("Account Verified?")
    submitted = st.form_submit_button("Submit")

# Load the SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Process the input and display the results
if submitted:
    new_data = process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status)
    
    if new_data is not None:
        st.write("Processed Data:", new_data)  # Debugging: Check the processed data
        st.write("Processed Data Shape:", new_data.shape)  # Debugging: Check the shape
        prediction = svm_model.predict(new_data)
        
        # Display prediction with custom styling
        st.markdown(
            f"""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; text-align: center;'>
                <h3 style='color: #2e7d32; font-family: Arial, sans-serif;'>
                    Predicted Misinformation Level: {prediction[0]}
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Error: Processed data is None. Check the process_tweet function.")

