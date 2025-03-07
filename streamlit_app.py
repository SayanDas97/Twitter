import streamlit as st

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

# Display the submitted data
if submitted:
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
