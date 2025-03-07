import streamlit as st
import pandas as pd
import re
import nltk
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
def process_tweet(tweet_body, joining_date, followers, followings):
    # Create a dictionary to store the results
    result = {}

    # Count the number of words in the tweet
    result['Word Count'] = len(str(tweet_body).split())

    # Count sensational words in the tweet
    result['Sensational words count'] = count_sensational_words(tweet_body)

    # Count consecutive capital words in the tweet
    result['Consecutive Capitals'] = sum(1 for word in str(tweet_body).split() if word.isupper() and len(word) > 1)

    # Count exclamation marks in the tweet
    result['Exclamation mark count'] = str(tweet_body).count('!')

    # Count question marks in the tweet
    result['Question mark count'] = str(tweet_body).count('?')

    # Count consecutive dots in the tweet
    result['incomplete sentence indicator'] = str(tweet_body).count('..')

    # Perform sentiment analysis
    result['Sentiment Score'] = vader.polarity_scores(tweet_body)['compound']
    result['Sentiment Category'] = sentiment_vader(tweet_body)

    # Count hashtags
    result['Hashtags used'] = count_hashtags(tweet_body)

    # Count emojis
    result['Number of emoticons'] = count_emojis(tweet_body)

    # Calculate Account Age
    joining_date = pd.to_datetime(joining_date)
    result['Account Age'] = (datetime.now() - joining_date).days

    # Count the number of @ symbols (mentions)
    result['mention count'] = str(tweet_body).count('@')

    return result

# Streamlit UI
st.title("Twitter Analytics Input Form")

# Create a form for user input
with st.form("twitter_form"):
    # Text input for Tweet Body
    tweet_body = st.text_area("Tweet Body", placeholder="Enter the tweet text here...")

    # Numeric inputs for engagement metrics
    likes = st.number_input("Likes", min_value=0, value=0)
    retweets = st.number_input("Retweets", min_value=0, value=0)
    comments = st.number_input("Comments", min_value=0, value=0)
    quotes = st.number_input("Quotes", min_value=0, value=0)
    views = st.number_input("Views", min_value=0, value=0)

    # Numeric inputs for follower/following counts
    followers = st.number_input("Number of Followers", min_value=0, value=0)
    followings = st.number_input("Number of Followings", min_value=0, value=0)

    # Date input for Date of Joining
    joining_date = st.date_input("Date of Joining")

    # Checkbox for Account Verification Status
    verified_status = st.checkbox("Account Verified?")

    # Submit button
    submitted = st.form_submit_button("Submit")

# Process the input and display the results
if submitted:
    # Process the tweet
    result = process_tweet(tweet_body, joining_date, followers, followings)

    # Display the results
    st.subheader("Analysis Results")
    st.write(f"**Word Count:** {result['Word Count']}")
    st.write(f"**Sensational words count:** {result['Sensational words count']}")
    st.write(f"**Consecutive Capitals:** {result['Consecutive Capitals']}")
    st.write(f"**Exclamation mark count:** {result['Exclamation mark count']}")
    st.write(f"**Question mark count:** {result['Question mark count']}")
    st.write(f"**Incomplete sentence indicator:** {result['incomplete sentence indicator']}")
    st.write(f"**Sentiment Score:** {result['Sentiment Score']}")
    st.write(f"**Sentiment Category:** {result['Sentiment Category']}")
    st.write(f"**Hashtags used:** {result['Hashtags used']}")
    st.write(f"**Number of emoticons:** {result['Number of emoticons']}")
    st.write(f"**Account Age:** {result['Account Age']}")
    st.write(f"**Mention count:** {result['mention count']}")
