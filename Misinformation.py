import streamlit as st
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load dataset
data = pd.read_csv("/mnt/data/tweet_data_v2.csv")
X = data[["Likes", "Retweets", "Comments", "Quotes", "Views", "Followers"]]  # Features from UI
y = data["Misinformation"]  # Target column

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Save model using pickle
with open("/mnt/data/svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

# Load model
def load_model():
    with open("/mnt/data/svm_model.pkl", "rb") as file:
        return pickle.load(file)

# Streamlit UI Integration
st.title("Twitter Engagement Prediction using SVM")

with st.form("twitter_form"):
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

if submitted:
    model = load_model()
    prediction = model.predict([[likes, retweets, comments, quotes, views, followers]])[0]
    st.subheader("Prediction Result")
    st.write(f"Predicted Misinformation Level: {prediction}")
