import pandas as pd
#load datset
df=pd.read_csv(r"E:\Datascience\Recommendation engine\Dataset (1)\Datasets_Recommendation Engine\game.csv")
#Data inspection
df.info()
df.describe()
df.isnull().sum()
#Created user item matrix for recommender system
user_item_matrix = df.pivot_table(index='userId', columns='game', values='rating')

# Fill missing values with 0 (for similarity computation)
user_item_matrix_filled = user_item_matrix.fillna(0)
print(user_item_matrix_filled.head())
#Exploratory data analysis
print("Shape:", df.shape)
print("\nSummary Statistics:")
print(df.describe())
print("\nTop Rated Games:")
print(df.groupby('game')['rating'].mean().sort_values(ascending=False).head(10))

#Univariate Analysis
import matplotlib.pyplot as plt

# Rating distribution
plt.hist(df['rating'], bins=10)
plt.title("Rating Distribution")
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.show()

# Top 10 most rated games
top_games = df['game'].value_counts().head(10)
top_games.plot(kind='bar')
plt.title("Top 10 Most Rated Games")
plt.xlabel("Game")
plt.ylabel("Number of Ratings")
plt.show()


# Average rating per game
avg_rating = df.groupby('game')['rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
avg_rating.head(10).plot(kind='bar')
plt.title("Top 10 Games by Average Rating")
plt.xlabel("Game")
plt.ylabel("Average Rating")
plt.show()

# User rating behavior
user_ratings = df.groupby('userId')['rating'].count()
plt.hist(user_ratings, bins=20)
plt.title("User Rating Frequency")
plt.xlabel("Number of Ratings per User")
plt.ylabel("Frequency")
plt.show()

from sklearn.metrics.pairwise import cosine_similarity

# Compute user similarity matrix
user_similarity = cosine_similarity(user_item_matrix_filled)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix_filled.index,
    columns=user_item_matrix_filled.index
)

# Function to recommend games using UBCF
def recommend_games(user_id, n_recommendations=5):
    # Get similarity scores for the user
    sim_scores = user_similarity_df[user_id].sort_values(ascending=False)
    
    # Remove the user itself
    sim_scores = sim_scores.drop(user_id)
    
    # Get most similar users
    similar_users = sim_scores.head(5).index
    
    # Get games rated by similar users
    similar_users_ratings = user_item_matrix.loc[similar_users]
    
    # Compute average rating from similar users
    recommended_games = similar_users_ratings.mean().sort_values(ascending=False)
    
    # Remove games already rated by the target user
    user_rated_games = user_item_matrix.loc[user_id].dropna().index
    recommended_games = recommended_games.drop(user_rated_games, errors='ignore')
    
    return recommended_games.head(n_recommendations)

# Example Recommendation
print("Recommended Games for User 3:")
print(recommend_games(3))

# Top selling (most frequently rated) DVDs
top_selling = df['game'].value_counts().head(10)
print("Top-Selling DVDs:")
print(top_selling)



