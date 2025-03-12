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
    try:
        # Debugging: Print input values
        print("Tweet Body:", tweet_body)
        print("Joining Date:", joining_date)
        print("Followers:", followers)
        print("Followings:", followings)
        print("Likes:", likes)
        print("Retweets:", retweets)
        print("Comments:", comments)
        print("Quotes:", quotes)
        print("Views:", views)
        print("Verified Status:", verified_status)

        # Count the number of words in the tweet
        word_count = len(str(tweet_body).split())

        # Count sensational words in the tweet
        sensational_words = count_sensational_words(tweet_body)

        # Count consecutive capital words in the tweet
        consecutive_capitals = sum(1 for word in str(tweet_body).split() if word.isupper() and len(word) > 2)

        # Count exclamation marks in the tweet
        exclamation_count = str(tweet_body).count('!')

        # Count question marks in the tweet
        question_count = str(tweet_body).count('?')

        # Count consecutive dots in the tweet
        incomplete_sentence_count = str(tweet_body).count('..')

        # Perform sentiment analysis
        simply_sentiment = vader.polarity_scores(tweet_body)['compound']
        Sentiment_Category = sentiment_vader(tweet_body)

        # Count hashtags
        hashtag_count = count_hashtags(tweet_body)
        
        # Count emojis
        emoji_count = count_emojis(tweet_body)
        
        # Calculate Account Age
        joining_date = pd.to_datetime(joining_date)
        age_in_days = (datetime.now() - joining_date).days
        
        # Count the number of @ symbols (mentions)
        mention_count = str(tweet_body).count('@')
        
        # Clickbait Score
        clickbait_usage = (
            exclamation_count + question_count + consecutive_capitals +
            incomplete_sentence_count + emoji_count + hashtag_count
        ) 
        Clickbait_Score = round(clickbait_usage / (1 + clickbait_usage), 4) if clickbait_usage > 0 else 0

        # Hyperbole Score
        Hyperbole_Score = round(sensational_words / (1 + sensational_words), 4) if sensational_words > 0 else 0
       
        # HC Sentiment
        HC_Sentiment = round(abs(simply_sentiment) * ((Clickbait_Score + Hyperbole_Score) / 2), 4)

        # Engagement Ratio
        engagement_ratio = (
            likes + retweets + comments + quotes
        ) / views if views > 0 else 0

        # HC Tweet Engagement Ratio
        HC_TER = round(engagement_ratio * (Clickbait_Score + Hyperbole_Score) / 2, 4)

        # Follower following ratio
        follower_following_ratio = (
            followers
        ) / followings if followings > 0 else 0

        # FFR to Misinfo
        U_Shaped_FFR = (
            round(1 - math.exp(-0.8 * abs(math.log(followers / followings))), 4)
            if (followers > 0 and followings > 0)
            else 1
        )
        # VA freshness score
        account_age_to_misinfo = round(math.exp(-0.01 * age_in_days), 4)
        VA_Freshness_Score = round(0.2 if verified_status == 1 else 0.8 * account_age_to_misinfo, 4)

        # Prepare the input for the SVM model
        input_data = pd.DataFrame({
            'Clickbait_Score': [Clickbait_Score],
            'Hyperbole_Score': [Hyperbole_Score],
            'HC_Sentiment': [HC_Sentiment],
            'HC_TER': [HC_TER],
            'U_Shaped_FFR': [U_Shaped_FFR],
            'VA_Freshness_Score': [VA_Freshness_Score],
        })        
        return input_data
    except Exception as e:
        st.error(f"Error in process_tweet: {e}")
        return None

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

