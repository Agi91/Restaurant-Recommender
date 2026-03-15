
import pandas as pd

# 1. Dataset load 
df = pd.read_csv("dataset.csv")  

# 2. check shape and colum of data set
print("Shape:", df.shape)
print("Columns:", df.columns)

# first 5 row
print(df.head())
# 4. Missing values check 

useful_cols = [
    "Restaurant Name", "City", "Locality", "Cuisines",
    "Average Cost for two", "Price range",
    "Has Table booking", "Has Online delivery",
    "Aggregate rating", "Votes"
]

data = df[useful_cols].copy()

# Missing values one step fill
fill_cols = ["Cuisines", "Locality", "City"]
data[fill_cols] = data[fill_cols].fillna("Unknown")

# Output
print("Missing values after cleaning:\n", data.isnull().sum())
print("Shape:", data.shape)
print(data.head())

# 1.make Price bucket 
def get_price_bucket(x):
    if pd.isna(x):
        return "mid"   # default
    if x == 1:
        return "budget"
    elif x == 2:
        return "mid"
    elif x == 3:
        return "premium"
    elif x == 4:
        return "luxury"
    else:
        return "mid"

data["Price_bucket"] = data["Price range"].apply(get_price_bucket)

# 2. Flags normalize  (Yes/No → text tokens)
data["Has Table booking"] = data["Has Table booking"].apply(
    lambda x: "table_booking" if str(x).strip().lower() in ["yes", "1", "true"] else ""
)

data["Has Online delivery"] = data["Has Online delivery"].apply(
    lambda x: "online_delivery" if str(x).strip().lower() in ["yes", "1", "true"] else ""
)

# Check output
print(data[["Restaurant Name", "Price range", "Price_bucket", "Has Table booking", "Has Online delivery"]].head(10))

cols_for_soup = ["Cuisines", "City", "Locality", "Price_bucket", 
                 "Has Table booking", "Has Online delivery"]

data["soup"] = data[cols_for_soup].astype(str).agg(" ".join, axis=1)

# Output check
print(data[["Restaurant Name", "soup"]].head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. TF-IDF Vectorizer lagao soup column 
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["soup"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 2. Cosine similarity calculate 
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Cosine similarity matrix shape:", cosine_sim.shape)

# Index mapping
indices = pd.Series(data.index, index=data["Restaurant Name"]).drop_duplicates()

def recommend_restaurants(name, top_n=10):
    if name not in indices:
        return f"'{name}' not found in dataset."
    
    idx = indices[name]
    
    # Similarity scores ko sort karke top_n pick karo (self ko skip karke)
    sim_scores = cosine_sim[idx].argsort()[::-1][1:top_n+1]
    
    #  required columns 
    return data.loc[sim_scores, 
        ["Restaurant Name", "City", "Locality", "Cuisines", "Price_bucket", "Aggregate rating", "Votes"]
    ]

print(recommend_restaurants("Ooma", top_n=5))

def recommend_by_preferences(cuisine=None, city=None, price=None, online=False, table=False, top_n=10):
    # Preference string 
    prefs = []
    if cuisine: prefs.append(cuisine)
    if city: prefs.append(city)
    if price: prefs.append(price)
    if online: prefs.append("online_delivery")
    if table: prefs.append("table_booking")

    if not prefs:
        return "⚠️ Please provide at least one preference!"

    query = " ".join(prefs)

    # Transform query into TF-IDF vector
    query_vec = vectorizer.transform([query])

    # Cosine similarity with all restaurants
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Top-N results
    top_indices = sim_scores.argsort()[::-1][:top_n]

    return data.loc[top_indices, 
        ["Restaurant Name", "City", "Locality", "Cuisines", "Price_bucket", "Aggregate rating", "Votes"]
    ]

print(recommend_by_preferences(
    cuisine="Japanese", 
    city="Mandaluyong City", 
    price="premium", 
    online=False, 
    table=False, 
    top_n=5
))


import numpy as np

def recommend_by_preferences_ranked(cuisine=None, city=None, price=None, online=False, table=False, top_n=10):
    # 1. Build preference query
    prefs = []
    if cuisine: prefs.append(cuisine)
    if city: prefs.append(city)
    if price: prefs.append(price)
    if online: prefs.append("online_delivery")
    if table: prefs.append("table_booking")

    if not prefs:
        return "⚠️ Please provide at least one preference!"

    query = " ".join(prefs)

    # 2. Convert query into TF-IDF vector
    query_vec = vectorizer.transform([query])

    # 3. Cosine similarity with all restaurants
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 4. Normalize ratings and votes
    ratings = data["Aggregate rating"].astype(float).fillna(0) / 5.0  # 0–1 scale
    votes = data["Votes"].astype(float).fillna(0)
    votes = np.log1p(votes) / np.log1p(votes.max())  # log scale normalize 0–1

    # 5. Weighted score
    final_score = (0.6 * sim_scores) + (0.3 * ratings) + (0.1 * votes)

    # 6. Get top-N
    top_indices = final_score.argsort()[::-1][:top_n]

    # 7. Return results
    result = data.loc[top_indices, 
        ["Restaurant Name", "City", "Locality", "Cuisines", "Price_bucket", "Aggregate rating", "Votes"]
    ].copy()
    result["Final Score"] = final_score[top_indices].round(3)

    return result












