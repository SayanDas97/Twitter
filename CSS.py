import streamlit as st
import pandas as pd
import re
import nltk
import math
import pickle
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (one-time only)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Define sensational keywords
sensational_keywords = [...]  # Same list as before

# Function to count sensational words in a tweet
def count_sensational_words(text):
    return sum(1 for word in sensational_keywords if word.lower() in text.lower())

# Function to count hashtags
def count_hashtags(text):
    return len(re.findall(r'#\w+', text))

# Function to count emojis
def count_emojis(text):
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(text))

# Function to determine sentiment
def sentiment_vader(text):
    polarity = vader.polarity_scores(text)['compound']
    return "positive" if polarity >= 0.05 else "negative" if polarity <= -0.05 else "neutral"

# Function to process a tweet
def process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status):
    sensational_words = count_sensational_words(tweet_body)
    exclamation_count = tweet_body.count('!')
    question_count = tweet_body.count('?')
    incomplete_sentence_count = tweet_body.count('..')
    hashtag_count = count_hashtags(tweet_body)
    emoji_count = count_emojis(tweet_body)
    sentiment_score = vader.polarity_scores(tweet_body)['compound']
    sentiment_category = sentiment_vader(tweet_body)
    joining_date = pd.to_datetime(joining_date)
    account_age_days = (datetime.now() - joining_date).days
    engagement_ratio = (likes + retweets + comments + quotes) / views if views > 0 else 0
    clickbait_score = round((exclamation_count + question_count + incomplete_sentence_count + emoji_count + hashtag_count) / 10, 4)
    hyperbole_score = round(sensational_words / 10, 4)
    hc_sentiment = round(abs(sentiment_score) * ((clickbait_score + hyperbole_score) / 2), 4)
    hc_ter = round(engagement_ratio * (clickbait_score + hyperbole_score) / 2, 4)
    u_shaped_ffr = round(1 - math.exp(-0.8 * abs(math.log(followers / followings))), 4) if (followers > 0 and followings > 0) else 1
    va_freshness_score = round(0.2 if verified_status else 0.8 * math.exp(-0.01 * account_age_days), 4)
    return pd.DataFrame({'Clickbait_Score': [clickbait_score], 'Hyperbole_Score': [hyperbole_score], 'HC_Sentiment': [hc_sentiment], 'HC_TER': [hc_ter], 'U_Shaped_FFR': [u_shaped_ffr], 'VA_Freshness_Score': [va_freshness_score]})

# Load the SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Streamlit UI with custom CSS
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f4;
    }
    .main {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stTextArea, .stNumberInput, .stDateInput, .stCheckbox, .stButton {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛑 Twitter Misinformation Detection")
st.subheader("Analyze tweets for potential misinformation indicators")

# Create a form for user input
with st.form("twitter_form"):
    tweet_body = st.text_area("Tweet Body", placeholder="Enter the tweet text here...")
    likes = st.number_input("Likes", min_value=0, value=0)
    retweets = st.number_input("Retweets", min_value=0, value=0)
    comments = st.number_input("Comments", min_value=0, value=0)
    quotes = st.number_input("Quotes", min_value=0, value=0)
    views = st.number_input("Views", min_value=0, value=0)
    followers = st.number_input("Followers", min_value=0, value=0)
    followings = st.number_input("Followings", min_value=0, value=0)
    joining_date = st.date_input("Joining Date")
    verified_status = st.checkbox("Verified Account")
    submitted = st.form_submit_button("Analyze Tweet")

if submitted:
    new_data = process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status)
    prediction = svm_model.predict(new_data)
    st.success(f"**Predicted Misinformation Level:** {prediction[0]}")
