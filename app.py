import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title(" Game Recommendation System (User-Based Collaborative Filtering)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"E:\Datascience\Recommendation engine\Recommendation engine\game.csv")
    return df

df = load_data()

# Show raw data
if st.checkbox("Show Raw Dataset"):
    st.write(df.head())

# Data Info
st.subheader("Dataset Information")
st.write("Shape:", df.shape)
st.write("Missing Values:")
st.write(df.isnull().sum())

# Create user-item matrix
user_item_matrix = df.pivot_table(index='userId', columns='game', values='rating')
user_item_matrix_filled = user_item_matrix.fillna(0)

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")

# Rating Distribution
st.write("### Rating Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(df['rating'], bins=10)
ax1.set_title("Rating Distribution")
ax1.set_xlabel("Ratings")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# Top 10 Most Rated Games
st.write("### Top 10 Most Rated Games")
top_games = df['game'].value_counts().head(10)
st.bar_chart(top_games)

# Top 10 Games by Average Rating
st.write("### Top 10 Games by Average Rating")
avg_rating = df.groupby('game')['rating'].mean().sort_values(ascending=False).head(10)
st.bar_chart(avg_rating)

# User rating behavior
st.write("### User Rating Frequency")
user_ratings = df.groupby('userId')['rating'].count()
fig2, ax2 = plt.subplots()
ax2.hist(user_ratings, bins=20)
ax2.set_title("User Rating Frequency")
ax2.set_xlabel("Number of Ratings per User")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# Compute similarity
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix_filled.index,
    columns=user_item_matrix_filled.index
)

# Recommendation function
def recommend_games(user_id, n_recommendations=5):
    if user_id not in user_item_matrix.index:
        return pd.Series(["User not found in dataset"])

    sim_scores = user_similarity_df[user_id].sort_values(ascending=False)
    sim_scores = sim_scores.drop(user_id)

    similar_users = sim_scores.head(5).index
    similar_users_ratings = user_item_matrix.loc[similar_users]

    recommended_games = similar_users_ratings.mean().sort_values(ascending=False)

    user_rated_games = user_item_matrix.loc[user_id].dropna().index
    recommended_games = recommended_games.drop(user_rated_games, errors='ignore')

    return recommended_games.head(n_recommendations)

# Sidebar for User Input
st.sidebar.header(" Get Recommendations")
user_id = st.sidebar.selectbox("Select User ID", user_item_matrix.index.tolist())
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Show Recommendations
st.subheader(" Recommended Games")
if st.button("Recommend"):
    recommendations = recommend_games(user_id, num_recommendations)
    st.write(recommendations)

# Top selling (most frequently rated games)
st.subheader(" Top-Selling (Most Rated) Games")
top_selling = df['game'].value_counts().head(10)
st.bar_chart(top_selling)