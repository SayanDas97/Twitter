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

# Define sensational keywords
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

# Function to count sensational words in a tweet
def count_sensational_words(text):
    return sum(1 for word in sensational_keywords if word.lower() in text.lower())

# Function to count hashtags
def count_hashtags(text):
    return len(re.findall(r'#\w+', text))

# Function to count emojis (basic approach)
def count_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B9"  # Misc symbols
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))

# Function to determine sentiment based on VADER compound score
def sentiment_vader(text):
    over_all_polarity = vader.polarity_scores(text)
    if over_all_polarity['compound'] >= 0.05:
        return "positive"
    elif over_all_polarity['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"
        
# Function to process a single tweet
def process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status):
    # Create a dictionary to store the results
    result = {}

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
    Clickbait_Score = round(clickbait_usage / (1 + clickbait_usage), 4)

    # Hyperbole Score
    Hyperbole_Score = round(sensational_words / (1 + sensational_words), 4)
   
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

    return result

# Streamlit UI
st.title("Twitter Misinformation Detection")

# Create a form for user input
with st.form("twitter_form"):
    # Input fields
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

    # Process the input and display the results
if submitted:
        # Process the tweet
        result = process_tweet(tweet_body, joining_date, followers, followings, likes, retweets, comments, quotes, views, verified_status)
# Load the SVM model
def load_model():
    with open('svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    return svm_model

with open('svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    
"""
data = {
    "Clickbait_Score": [0.3],
    "Hyperbole_Score": [0.8], 
    "HC_Sentiment": [0.6],
    "HC_TER": [0.9],
    "U_Shaped_FFR": [0.3],
    "VA_Freshness_Score": [0.8]
}
df_data = pd.DataFrame(data)
        # Make prediction using the SVM model
#svm_model = load_model()
"""
# Prepare the input for the SVM model
input_data = pd.DataFrame({
    "Clickbait_Score" : Clickbait_Score,
    "Hyperbole_Score": Hyperbole_Score,
    "HC_Sentiment": HC_Sentiment,
    "HC_TER": HC_TER,
    "U_Shaped_FFR": U_Shaped_FFR,
    "VA_Freshness_Score": VA_Freshness_Score
})
prediction = svm_model.predict(df_data)
st.write(prediction)

# Map the prediction to the corresponding class label
class_labels = ['Low', 'No', 'Moderate', 'High']
predicted_class = class_labels[prediction[0]]

# Display the prediction result with class levels
st.subheader("Misinformation Prediction")
# st.write(f"**Predicted Misinformation Level:** {predicted_class}")


