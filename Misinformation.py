import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Title of the app
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

# Display the submitted data and make prediction
if submitted:
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Likes': [likes],
        'Retweets': [retweets],
        'Comments': [comments],
        'Quotes': [quotes],
        'Views': [views],
        'Followers': [followers],
        'Followings': [followings],
        'Verified': [1 if verified_status else 0],
        # Add other features as needed
    })

    # Make prediction using the loaded model
    prediction = svm_model.predict(input_data)

    # Map the prediction to the corresponding class label
    class_labels = ['Low', 'No', 'Moderate', 'High', 'Severe']
    predicted_class = class_labels[prediction[0]]

    # Display the submitted data and prediction
    st.subheader("Submitted Data")
    st.write(f"**Tweet Body:** {tweet_body}")
    st.write(f"**Likes:** {likes}")
    st.write(f"**Retweets:** {retweets}")
    st.write(f"**Comments:** {comments}")
    st.write(f"**Quotes:** {quotes}")
    st.write(f"**Views:** {views}")
    st.write(f"**Number of Followers:** {followers}")
    st.write(f"**Number of Followings:** {followings}")
    st.write(f"**Date of Joining:** {joining_date}")
    st.write(f"**Account Verified:** {'Yes' if verified_status else 'No'}")
    st.write(f"**Predicted Misinformation Class:** {predicted_class}")
    prediction = model.predict([[likes, retweets, comments, quotes, views, followers]])[0]
    st.subheader("Prediction Result")
    st.write(f"Predicted Misinformation Level: {prediction}")
