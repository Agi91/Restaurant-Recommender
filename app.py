import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load trained model (.pkl)
# -----------------------------
@st.cache_resource
def load_model():
    with open("restaurant_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

vectorizer = model["vectorizer"]
tfidf_matrix = model["tfidf_matrix"]
cosine_sim = model["cosine_sim"]
data = model["data"]

# -----------------------------
# Recommendation functions
# -----------------------------
indices = data.reset_index().set_index("Restaurant Name")["index"]

def recommend_restaurants(name, top_n=10):

    if name not in indices:
        return f"'{name}' not found in dataset."

    idx = indices[name]

    sim_scores = cosine_sim[idx].argsort()[::-1][1:top_n+1]

    return data.loc[sim_scores,
        ["Restaurant Name","City","Locality","Cuisines","Price_bucket","Aggregate rating","Votes"]
    ]


def recommend_by_preferences_ranked(cuisine=None, city=None, price=None, online=False, table=False, top_n=10):

    prefs = []

    if cuisine:
        prefs.append(cuisine)

    if city:
        prefs.append(city)

    if price:
        prefs.append(price)

    if online:
        prefs.append("online_delivery")

    if table:
        prefs.append("table_booking")

    query = " ".join(prefs)

    query_vec = vectorizer.transform([query])

    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    ratings = data["Aggregate rating"].astype(float).fillna(0) / 5.0

    votes = data["Votes"].astype(float).fillna(0)
    votes = np.log1p(votes) / np.log1p(votes.max())

    final_score = (0.6 * sim_scores) + (0.3 * ratings) + (0.1 * votes)

    top_indices = final_score.argsort()[::-1][:top_n]

    result = data.loc[top_indices,
        ["Restaurant Name","City","Locality","Cuisines","Price_bucket","Aggregate rating","Votes"]
    ].copy()

    result["Final Score"] = final_score[top_indices].round(3)

    return result


# -----------------------------
# UI START (same as yours)
# -----------------------------

st.set_page_config(page_title="Restaurant Recommender", layout="wide", page_icon="🍽")

st.title("🍽 Restaurant Recommendation System")
st.write("Find restaurants based on your preferences")

st.sidebar.header("Your Preferences")
st.sidebar.markdown(
    """
    Use the filters below to find restaurants that match your tastes and needs!
    """
)

# -----------------------------
# Dropdown options from dataset
# -----------------------------
cities = sorted(data["City"].dropna().unique())
cuisines = sorted(data["Cuisines"].dropna().unique())

# Sidebar Filters
cuisine = st.sidebar.selectbox(
    "Cuisine (e.g., Japanese)",
    [""] + list(cuisines)
)

city = st.sidebar.selectbox(
    "City",
    [""] + list(cities)
)

price = st.sidebar.selectbox(
    "Price Range",
    ["", "budget", "mid", "premium", "luxury"]
)

online = st.sidebar.checkbox("Online Delivery")
table = st.sidebar.checkbox("Table Booking")

top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

# Sidebar Button
if st.sidebar.button("Recommend Restaurants"):

    results = recommend_by_preferences_ranked(
        cuisine=cuisine if cuisine else None,
        city=city if city else None,
        price=price if price else None,
        online=online,
        table=table,
        top_n=top_n
    )

    st.subheader("Recommended Restaurants")

    if isinstance(results, str):
        st.warning(results)
    else:
        st.dataframe(results, use_container_width=True)


st.markdown("---")

# -----------------------------
# Find Similar Restaurants
# -----------------------------
st.subheader("Find Similar Restaurants")

restaurant_list = sorted(data["Restaurant Name"].unique())

restaurant_name = st.selectbox(
    "Enter Restaurant Name",
    [""] + restaurant_list
)

if st.button("Find Similar"):

    results = recommend_restaurants(restaurant_name)

    if isinstance(results, str):
        st.warning(results)
    else:
        st.dataframe(results, use_container_width=True)


# -----------------------------
# SAME CSS (unchanged)
# -----------------------------
st.markdown(
    """
    <style>
        .css-1d391kg {
            background-color: #f5f5f5;
        }
        
        .css-1v0mbdj {
            background-color: #008b8b;
            color: white;
        }
        
        .css-1g1j4cx {
            background-color: #008b8b;
            color: white;
        }

        .stButton>button {
            background-color: #008b8b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }

        .stButton>button:hover {
            background-color: #006666;
        }

        .stSelectbox, .stTextInput, .stSlider, .stCheckbox {
            font-size: 14px;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }

        h1 {
            color: #008b8b;
        }

        h3 {
            color: #555;
        }

        .css-ffhzg2 {
            font-weight: bold;
            color: #008b8b;
        }
    </style>
    """,
    unsafe_allow_html=True
)