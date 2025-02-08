# NEET Student Rank Predictor

## Project Overview

The **NEET Student Rank Predictor** is a web application built using **Streamlit**, designed to predict the NEET (National Eligibility cum Entrance Test) rank and suggest potential colleges based on a student's performance. It leverages historical student data to predict future ranks using machine learning models (Linear Regression) and provides a detailed analysis of student performance. This analysis includes visualizations that allow users to explore their score, accuracy, and weak areas.

### Key Features:
- **Rank Prediction**: Predicts NEET rank based on user performance.
- **College Prediction**: Suggests the most likely college based on predicted rank.
- **Performance Analysis**: Visualizes the student's performance over time, showing score and accuracy trends.
- **Weak Areas Analysis**: Highlights the student's weakest topics based on their accuracy.
- **Interactive Visualizations**: Includes interactive charts for better user engagement and data exploration.

## Approach

1. **Data Collection**: 
   - The app uses data from multiple API endpoints to fetch the student's quiz history, latest submissions, and NEET cutoff ranks of various medical colleges.
   
2. **Data Preprocessing**:
   - Data is cleaned and formatted into a DataFrame for easy analysis and manipulation.
   - The accuracy values are cleaned and converted into numeric format for model training.

3. **Rank Prediction Model**:
   - A Linear Regression model is used to predict the NEET rank based on the student’s quiz scores, accuracy, and final scores.
   
4. **College Prediction**:
   - Based on the predicted rank, the app suggests a college from a predefined list of medical colleges.

5. **Data Visualization**:
   - Various interactive charts are displayed, including line charts (for score and accuracy trends) and bar charts (for score vs accuracy), making it easy to interpret the student's progress.
   - A heatmap is also provided to analyze the weak areas of the student’s performance.

6. **User Interface**:
   - The app is built using **Streamlit** for rapid development and interactive elements, allowing users to adjust their scores and accuracy, and view predictions and insights in real-time.

## Setup Instructions

### Prerequisites
Ensure that you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)


