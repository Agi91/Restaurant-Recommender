<<<<<<< HEAD

````markdown
# 🍽 AI Restaurant Recommender

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

# 📌 Project Overview

**AI Restaurant Recommender** is a Machine Learning project designed to recommend restaurants based on user preferences such as **cuisine, city, locality, price range, table booking, and online delivery availability**.  

The project demonstrates a complete **end-to-end recommendation workflow**, including:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Content-based similarity computation
- Interactive dashboard deployment

An interactive **Streamlit dashboard** allows users to select preferences or search for a restaurant by name and instantly get **ranked restaurant recommendations**.

This project showcases practical skills in **machine learning, natural language processing, content-based filtering, and building interactive web applications**.

---

# ❓ Key Questions Explored

- Which restaurants match a user’s **cuisine and city preference**?  
- How do **price range and service options** affect recommendations?  
- Which **highly-rated or popular restaurants** are similar to a selected restaurant?  
- Can we provide **ranked recommendations** combining similarity, ratings, and votes?  
- How do **online delivery and table booking** influence restaurant selection?  

---

# 🛠 Technologies & Tools Used

| Tool              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| Python            | Core programming language                                 |
| Pandas            | Data processing & preprocessing                           |
| NumPy             | Numerical computations                                    |
| Scikit-learn      | TF-IDF vectorization & similarity computation             |
| Streamlit         | Interactive web dashboard                                 |
| Pickle            | Save/load trained model                                   |
| Git & GitHub      | Version control & portfolio showcase                      |

---

# 📊 Dataset

**File in Repository:** `dataset.csv`  

The dataset contains information about restaurants including location, cuisine, pricing, services, customer votes, and aggregate ratings.

### Important Features

- Restaurant Name  
- City  
- Locality  
- Cuisines  
- Price Range / Price Bucket  
- Has Table Booking  
- Has Online Delivery  
- Aggregate Rating  
- Votes  

These features are used to **compute content similarity** and generate ranked restaurant recommendations.

---

# 📂 Project Structure

```bash
AI-Restaurant-Recommender/
│
├── dataset.csv
├── train_model.py
├── restaurant_model.pkl
├── recommender.py
├── app.py
├── requirements.txt
└── README.md
````

### Folder Details

* **dataset.csv**
  Restaurant dataset used for building the recommendation model.

* **train_model.py**
  Script used for preprocessing data, creating TF-IDF vectors, computing similarity, and saving the Pickle model.

* **restaurant_model.pkl**
  Saved pre-trained model for fast recommendations.

* **recommender.py**
  Contains functions: `recommend_restaurants` and `recommend_by_preferences_ranked`.

* **app.py**
  Streamlit web application for user interaction and recommendations.

* **requirements.txt**
  Python dependencies required to run the project.

---

# 🤖 Machine Learning Model

The system uses **content-based filtering** with TF-IDF vectorization of combined restaurant features (cuisines, city, locality, price bucket, and service options).

### Model Details

* Cosine similarity computes similarity between restaurants
* **Weighted ranking** combines:

  * 60% content similarity
  * 30% normalized aggregate rating
  * 10% log-scaled votes

This ensures **high-quality, ranked recommendations** rather than only similar items.

The trained model is saved as:

```bash
restaurant_model.pkl
```

and loaded in the Streamlit app for real-time recommendations.

---

# 🌐 Features of the Web Application

The Streamlit dashboard allows users to:

* Select **Cuisine** and **City** via dropdowns
* Choose **Price Range** (Budget / Mid / Premium / Luxury)
* Enable **Table Booking** and **Online Delivery** options
* Specify **number of recommendations**
* Search for similar restaurants by **restaurant name**

### Dashboard Highlights

* ⭐ Ranked recommendations based on similarity, ratings, and votes
* 🍔 Restaurant result cards
* 📊 Interactive charts and insights
* Fast recommendations using **pre-trained Pickle model**

---

# ▶ Running the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Agi91/Restaurant-Recommender.git
```

### 2️⃣ Navigate to the Project Folder

```bash
cd AI-Restaurant-Recommender
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

### 5️⃣ Open in Browser

```bash
http://localhost:8501
```

You will see the **Restaurant Recommender Dashboard**.

---

# 🚀 Future Improvements

* Improve dashboard UI with restaurant cards, ratings, and badges
* Integrate real-time restaurant APIs for dynamic data
* Add deep learning models for advanced recommendations
* Deploy on Streamlit Cloud or HuggingFace Spaces
* Include user reviews and sentiment analysis

---

# 👨‍💻 Author

**AI Restaurant Recommender**
Machine Learning & Web App Project

GitHub Repository:
[https://github.com/Agi91/Restaurant-Recommender](https://github.com/Agi91/Restaurant-Recommender)

⭐ Built with **Python, Machine Learning, and Streamlit**
=======
