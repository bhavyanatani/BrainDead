# Problem Statement:1 🎬 ReelSense: Explainable Movie Recommender System with Diversity Optimization hello


## Deployed Dashboard: https://reelsensemovie.streamlit.app

ReelSense is a hybrid movie recommendation system built using the **MovieLens Latest Small** dataset.  
The system generates **personalized Top-K recommendations**, provides **human-readable explanations**, and evaluates performance not only on ranking quality but also on **diversity and catalog coverage**.

All experiments, models, and evaluations are implemented end-to-end in a single Jupyter Notebook.

---

## 📁 Dataset

- **MovieLens Latest Small**
- 100,836 ratings from 610 users on 9,742 movies

### Files Used
- `ratings.csv` – user–movie ratings with timestamps  
- `movies.csv` – movie titles and genres  
- `tags.csv` – user-generated tags  
- `links.csv` – external identifiers  

---

## 🧹 Data Preparation

The notebook performs the following preprocessing steps:

- Time-aware train/test split (user-wise, last-N interactions held out)
- Parsing and normalization of movie genres
- Cleaning and aggregation of user tags
- Construction of user–item interaction data
- Feature preparation for both collaborative and content-based models

---

## 🔍 Exploratory Data Analysis

EDA is carried out to understand user and item behavior, including:

- Rating distribution across users and movies
- User activity levels
- Genre popularity patterns
- Long-tail characteristics of movie consumption
- Temporal trends in rating behavior

These insights guided model selection and evaluation strategy.

---

## 🧠 Recommendation Approaches Implemented

### 1️⃣ Popularity-Based Baseline
- Most-rated and highest-rated movies
- Used as a non-personalized reference baseline

### 2️⃣ Collaborative Filtering
- User–User similarity
- Item–Item similarity
- Recommendations generated from neighborhood-based similarity matrices

### 3️⃣ Matrix Factorization
- Singular Value Decomposition (SVD)
- Implemented using the **Surprise** library
- Trained on the user–item rating matrix

### 4️⃣ Hybrid Recommendation Model
- Combines:
  - Collaborative filtering scores
  - Content similarity from genres and tags
- Helps reduce popularity bias and improve coverage

---

## ✨ Explainable Recommendations

For each recommended movie, the system generates **natural language explanations** based on:

- Overlapping genres
- Shared user tags
- Similarity to movies previously rated by the user

### Example Explanation
> *Recommended because you liked movies with similar genres and tags such as sci-fi and mind-bending themes.*

This makes the recommendations transparent and interpretable.

---

## 🎯 Evaluation Metrics

The notebook evaluates performance using multiple perspectives:

### 🔹 Ranking Quality (Top-K)
- Precision@K  
- Recall@K  
- NDCG@K  
- MAP@K  

### 🔹 Rating Prediction (Matrix Factorization)
- RMSE  
- MAE  

### 🔹 Diversity & Novelty
- Catalog Coverage  
- Intra-List Diversity (ILD)  
- Popularity-aware recommendation analysis  

These metrics ensure the system balances **accuracy, personalization, and diversity**.

---

## 📊 Results Summary

- Hybrid recommendations outperform simple popularity-based approaches
- Collaborative filtering captures personalized preferences effectively
- Content features improve diversity and reduce over-recommendation of popular items
- Explainability layer provides meaningful justification for recommendations
- Diversity-aware evaluation shows improved catalog utilization

Detailed metric computations and outputs are available in the notebook.

---
