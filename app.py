import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

# Load JSON data from endpoints
def load_json(url):
    response = requests.get(url)
    return response.json()

# API Endpoints
quiz_url = "https://www.jsonkeeper.com/b/LLQT"
history_url = "https://api.jsonserve.com/XgAgFJ"
latest_submission_url = "https://api.jsonserve.com/rJvd7g"

# Hardcoded College Cutoff Data
college_cutoffs = [
    {"college_name": "KES College", "cutoff_rank": 100},
    {"college_name": "JIPMER, Puducherry", "cutoff_rank": 500},
    {"college_name": "Maulana Azad Medical College, Delhi", "cutoff_rank": 2000},
    {"college_name": "King George's Medical University, Lucknow", "cutoff_rank": 5000},
    {"college_name": "State Govt. Medical College", "cutoff_rank": 10000}
]

# Load data
quiz_data = load_json(quiz_url)
history_data = load_json(history_url)
latest_submission = load_json(latest_submission_url)

# Convert to DataFrames
quiz_df = pd.DataFrame(quiz_data['quiz']['questions'])
history_df = pd.DataFrame(history_data)
latest_df = pd.DataFrame([latest_submission])
college_df = pd.DataFrame(college_cutoffs)

# Extract useful columns
history_df = history_df[['user_id', 'score', 'accuracy', 'final_score', 'negative_score', 'correct_answers', 'incorrect_answers', 'better_than', 'rank_text']]

# Clean accuracy column
history_df['accuracy'] = history_df['accuracy'].str.replace('%', '').astype(float)

# Ensure that the latest submission's accuracy is properly cleaned
latest_df['accuracy'] = latest_df['accuracy'].str.replace('%', '').astype(float)

# Compute Rank Prediction Model
def predict_rank(user_id, modified_score=None, modified_accuracy=None):
    user_history = history_df[history_df['user_id'] == user_id]
    if user_history.empty:
        return "No sufficient data for rank prediction"

    # Ensure accuracy column is numeric
    user_history['accuracy'] = user_history['accuracy'].astype(float)

    X = user_history[['score', 'accuracy', 'final_score']].values
    y = user_history['better_than'].values

    model = LinearRegression()
    model.fit(X, y)

    # Get the latest submission scores and ensure accuracy is numeric
    latest_scores = latest_df[['score', 'accuracy', 'final_score']].values

    if modified_score:
        latest_scores[0][0] = modified_score
    if modified_accuracy:
        latest_scores[0][1] = modified_accuracy

    predicted_rank = model.predict(latest_scores)[0]
    return max(1, int(predicted_rank))  # Ensure rank is at least 1

# Hardcoded College Prediction based on Rank
def predict_college(rank):
    eligible_colleges = [college for college in college_cutoffs if college['cutoff_rank'] >= rank]
    if eligible_colleges:
        return eligible_colleges[0]['college_name']
    return "No eligible college found"

# Customizing theme
st.set_page_config(page_title="NEET Rank Predictor - KES College", layout="wide")
st.markdown("""
    <style>
        body {background-color: #f5f5f5; color: #333;}
        .stSidebar {background-color: #ffffff !important; box-shadow: 2px 0px 10px rgba(0, 0, 0, 0.1);}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px;}
        .stSlider>div>div>div>div {background-color: #4CAF50;}
        .css-1d391kg {padding: 20px; border-radius: 10px; background-color: #ffffff; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);}
        .css-1d391kg h1 {color: #4CAF50;}
        .css-1d391kg h2 {color: #333;}
        .css-1d391kg h3 {color: #555;}
        .stProgress>div>div>div>div {background-color: #4CAF50;}
    </style>
""", unsafe_allow_html=True)

# Streamlit UI with updated layout
st.title("🎓 NEET Rank Predictor - KES College")
st.sidebar.header("📊 User Selection")
selected_user = st.sidebar.selectbox("Select User ID", history_df['user_id'].unique())

# Updated sidebar UI with sliders and new layout
st.sidebar.subheader("📝 Modify Your Marks")
modified_score = st.sidebar.slider("Adjust Score", min_value=0, max_value=720, value=int(latest_df['score'][0]))
modified_accuracy = st.sidebar.slider("Modify Accuracy (%)", min_value=0, max_value=100, value=int(latest_df['accuracy'][0]))

# Rank Prediction
predicted_rank = predict_rank(selected_user, modified_score, modified_accuracy)
college_prediction = predict_college(predicted_rank)

# Updated UI alignment
with st.container():
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader(f"📈 Predicted NEET Rank: **{predicted_rank}**")
    with col2:
        st.subheader(f"🏫 Most Likely College: **{college_prediction}**")

# Progress Bar for Rank
st.subheader("📊 Rank Progress")
rank_progress = (1 - (predicted_rank / 10000)) * 100  # Assuming max rank is 10,000
st.progress(int(rank_progress))
st.caption(f"You are in the top **{predicted_rank}** ranks out of 10,000.")

# Updated Graph UI
st.subheader("📉 Performance Analysis")
user_data = history_df[history_df['user_id'] == selected_user]

col1, col2 = st.columns(2)
with col1:
    fig1 = px.line(user_data, x="rank_text", y=["score", "accuracy"], 
                   title="Score and Accuracy Over Time", labels={"rank_text": "Rank Text"}, color_discrete_sequence=["#4CAF50", "#FF5722"])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    chart = alt.Chart(user_data).mark_bar().encode(
        x=alt.X('rank_text:N', title='Rank Text'),
        y=alt.Y('score:Q', title='Score'),
        color='accuracy:Q',
        tooltip=['score', 'accuracy']
    ).properties(title="Score vs Accuracy by Rank")
    st.altair_chart(chart, use_container_width=True)

# Updated heatmap UI
st.subheader("📊 Weak Areas Analysis")
weak_topics = user_data.sort_values('accuracy').head(3)['rank_text']
st.write("Weak Topics:", weak_topics.values)
accuracy_matrix = user_data.pivot_table(index='rank_text', values='accuracy', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(accuracy_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Updated loading system
with st.spinner("🔍 Processing your request..."):
    st.success("✅ Analysis Completed Successfully!")