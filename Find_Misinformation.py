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
sensational_keywords =[
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
]  # Same list as before

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
st.markdown(
     <style>
    /* Background Styling */
    body {
        background-color: #D3D3D3;
        font-family: Arial, sans-serif;
    }

    /* Title Styling */
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 50px;
    
    }

    /* Headers */
    h1, h2, h3 {
        color: black;
        font-weight: bold;
        font-size: 30px;
        text-align: center;
    }

    /* Input Box Styling */
    textarea, input[type="number"], input[type="date"] {
        border: 2px solid black;
        font-weight: bold;
        color: white;
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
     label {
        font-size:30px !important;
        font-weight:bold !important;
        color:white !important;
    }
    </style>
    ,
    unsafe_allow_html=True
)



st.title("🛑Misinformation Detection in a tweet")
st.subheader("Analyze tweets for potential misinformation indicators")

# Create a form for user input
with st.form("twitter_form"):
    tweet_body = st.text_area("📝 Tweet Body", placeholder="Enter the tweet text here...")
    likes = st.number_input("❤️ Likes", min_value=0, value=0)
    retweets = st.number_input("🔁 Retweets", min_value=0, value=0)
    comments = st.number_input("💬 Comments", min_value=0, value=0)
    quotes = st.number_input("📢 Quotes", min_value=0, value=0)
    views = st.number_input("👀 Views", min_value=0, value=0)
    followers = st.number_input("👥 Number of Followers", min_value=0, value=0)
    followings = st.number_input("👤 Number of Followings", min_value=0, value=0)
    joining_date = st.date_input("📅 Date of Joining")
    verified_status = st.checkbox("Account Verified?")
    submitted = st.form_submit_button("🚀 Submit")

if submitted:
    new_data = process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status)
    prediction = svm_model.predict(new_data)
    st.success(f"**Predicted Misinformation Level:** {prediction[0]}")
