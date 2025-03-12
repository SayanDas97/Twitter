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

sensational_keywords = [
    "Unprecedented", "Biggest", "Worst", "Best", "Regret later",
    "Happening Now", "Before it too late", "Act now", "Wake up",
    "Must read", "Cover Up", "OMG", "Shocking", "Warning", "Urgent",
    "Breaking", "Exposed", "Unbelievable", "Disaster", "Deadly",
    "Emergency", "Dangerous", "Dirty", "Betrayal", "Lies", "Rigged", "Hoax",
    "Scam", "Fraud", "Corrupt", "Evil", "Agenda", "Experts say", "Scientists confirm",
    "They admit", "Whistleblower reveals", "Leaked documents show", "Insider knowledge",
    "You won’t believe", "They don’t want you to know", "Share before it’s deleted",
    "This will change everything", "Spread the truth", "Hidden", "Terrifying",
    "Catastrophe", "Collapse", "Outbreak", "Cover-up", "Secret", "Banned", "Suppressed",
    "Silenced", "Shadow government", "Deep state", "New World Order", "encouraged", "Collapse imminent",
    "On the verge of disaster", "Doomsday", "Endgame", "Final warning", "Tipping point", "The truth is out", "Mass panic", "Hidden crisis",
    "Cover story", "Top-secret", "What they’re hiding",
    "Little-known facts", "They don’t want you to see this", "Revealed at last", "It’s worse than you think", "History repeating itself",
    "A must-watch", "Never seen before", "Think for yourself", "Judge for yourself", "Doctors confirm", "Military insider leaks",
    "Respected sources say", "Government insider speaks out", "Hidden in plain sight", "Uncensored truth", "Exclusive access",
    "Classified documents reveal", "Shocking footage", "Verified proof", "They knew all along", "Not a coincidence", "Think twice",
    "Planned all along", "Silent war", "Hidden messages", "They control everything", "You’re being watched", "False flag", "Engineered crisis", "Time is running out",
    "Do this now", "You must act", "They will regret this", "The world is waking up", "Stand up before it’s too late", "Fight back",
    "Don't be fooled", "The moment of truth", "Expose the lies"
]

# Apply Custom CSS to Streamlit App
st.markdown(
    """
    <style>
    /* Background Styling */
    body {
        background-color: #f4f4f4;
        font-family: Arial, sans-serif;
    }

    /* Title Styling */
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }

    /* Headers */
    h1, h2, h3 {
        color: black;
        font-weight: bold;
        text-align: center;
    }

    /* Input Box Styling */
    textarea, input[type="number"], input[type="date"] {
        border: 2px solid black;
        font-weight: bold;
        color: black;
    }

    /* Submit Button Styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    
    /* Hover Effect on Button */
    .stButton>button:hover {
        background-color: #ff1c1c;
        transform: scale(1.05);
    }

    /* Result Styling */
    .result-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid black;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        text-align: center;
        font-weight: bold;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🚀 Twitter Misinformation Detection")

# Create a form for user input
with st.form("twitter_form"):
    # Input fields
    tweet_body = st.text_area("📝 Tweet Body", placeholder="Enter the tweet text here...")
    likes = st.number_input("❤️ Likes", min_value=0, value=0)
    retweets = st.number_input("🔁 Retweets", min_value=0, value=0)
    comments = st.number_input("💬 Comments", min_value=0, value=0)
    quotes = st.number_input("📢 Quotes", min_value=0, value=0)
    views = st.number_input("👀 Views", min_value=0, value=0)
    followers = st.number_input("👥 Number of Followers", min_value=0, value=0)
    followings = st.number_input("👤 Number of Followings", min_value=0, value=0)
    joining_date = st.date_input("📅 Date of Joining")
    verified_status = st.checkbox("✅ Account Verified?")
    submitted = st.form_submit_button("🚀 Submit")

# Load the SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Process the input and display the results
if submitted:
    new_data = process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status)
    prediction = svm_model.predict(new_data)

    st.subheader("🔍 Misinformation Prediction")
    st.markdown(f'<div class="result-box">Predicted Misinformation Level: <strong>{prediction[0]}</strong></div>', unsafe_allow_html=True)
