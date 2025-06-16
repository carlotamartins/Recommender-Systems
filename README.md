
# CineMatch 🎬 – Movie Recommendation System

CineMatch is a comprehensive movie recommendation system that explores, compares, and deploys various recommendation algorithms to deliver accurate and personalized movie suggestions. This project is designed for both new users and returning users, offering tailored experiences based on user history and preferences.

## 🚀 Project Overview

This project implements and evaluates several recommender system approaches:

- **Non-personalized recommendations** based on popularity (weighted rating)
- **Collaborative filtering** (item-based and user-based)
- **Matrix factorization** using Singular Value Decomposition (SVD)
- **Content-based filtering** using genres
- **Neural Collaborative Filtering (NCF)**
- **Hybrid recommendation model** combining collaborative and latent factor techniques

Each model is evaluated using precision@k and recall@k metrics.

## 🛠 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, TensorFlow)
- Streamlit (for deployment)
- Jupyter Notebooks (for EDA, modeling, and evaluation)
- Pickle (for saving model assets)

## 📁 Project Structure

```bash
├── script.ipynb/               # Jupyter notebooks for each algorithm
├── ml-latest-small/            # CSV files
│   └── movies.csv
│   └── ratings.csv
├── streamlit_files/           # Pickled data and models
│   └── hybrid_model_function.pkl  
│   └── recommender_assets.pkl  
├── functions.py               # Reusable scoring and recommendation functions
├── app.py                     # Streamlit app entry point
└── README.md
```

## 🧠 Models Overview

- **Collaborative Filtering:** Uses user-item interaction matrix and similarity metrics (cosine similarity).
- **Matrix Factorization (SVD):** Captures latent features using decomposition of the rating matrix.
- **Content-Based:** Recommends movies similar to what a user liked based on genre vectors.
- **Neural Network:** Predicts ratings using embeddings for users and movies trained through a deep neural architecture.
- **Hybrid:** Merges strengths of item-based, user-based, and matrix factorization models with tunable weights.

## 📊 Evaluation

Models are evaluated using:
- **Precision@k**: Proportion of recommended items that are relevant
- **Recall@k**: Proportion of relevant items that were recommended

## 💻 Running the App

To launch the Streamlit app locally:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cinematch.git
cd cinematch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```
