import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dataset load
df = pd.read_csv("dataset.csv")

useful_cols = [
    "Restaurant Name", "City", "Locality", "Cuisines",
    "Average Cost for two", "Price range",
    "Has Table booking", "Has Online delivery",
    "Aggregate rating", "Votes"
]

data = df[useful_cols].copy()

fill_cols = ["Cuisines", "Locality", "City"]
data[fill_cols] = data[fill_cols].fillna("Unknown")

# price bucket
def get_price_bucket(x):
    if x == 1:
        return "budget"
    elif x == 2:
        return "mid"
    elif x == 3:
        return "premium"
    elif x == 4:
        return "luxury"
    return "mid"

data["Price_bucket"] = data["Price range"].apply(get_price_bucket)

# normalize flags
data["Has Table booking"] = data["Has Table booking"].apply(
    lambda x: "table_booking" if str(x).lower()=="yes" else ""
)

data["Has Online delivery"] = data["Has Online delivery"].apply(
    lambda x: "online_delivery" if str(x).lower()=="yes" else ""
)

cols_for_soup = ["Cuisines","City","Locality","Price_bucket",
                 "Has Table booking","Has Online delivery"]

data["soup"] = data[cols_for_soup].astype(str).agg(" ".join,axis=1)

# TFIDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["soup"])

# cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# save model
model_data = {
    "vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix,
    "cosine_sim": cosine_sim,
    "data": data
}

with open("restaurant_model.pkl","wb") as f:
    pickle.dump(model_data,f)

print("Model saved successfully")