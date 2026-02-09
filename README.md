# Problem Statement:1 🎬 ReelSense: Explainable Movie Recommender System with Diversity Optimization 

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

# Problem Statement:2 Cognitive Radiology Report Generation

## Report (IEEE Format): https://github.com/bhavyanatani/BrainDead/blob/main/IEEE%20Format%20Report.pdf

## Demo Video: https://drive.google.com/drive/folders/1BWmLPHr8yTo1CCCFSXiv_AyglGLNK3fu

CogRRG is a **cognitively inspired AI “Second Reader” system** for **structured Chest X-Ray (CXR) report generation**, designed to reduce reader fatigue and improve clinical accuracy by explicitly simulating radiological reasoning.

This repository contains the official implementation of **CogRRG**, as described in our IEEE-format report:

> **CogRRG: A Swin-Base Cognitive Second-Reader for Structured Chest X-Ray Report Generation**

---

## 🚨 Motivation

Radiology reading rooms operate under intense cognitive load, leading to **3–5% discrepancy rates** in human-generated reports due to reader fatigue.  
Most existing AI systems treat report generation as an **image captioning problem**, achieving good BLEU/CIDEr scores while often **hallucinating clinical findings**.

CogRRG reframes report generation as a **cognitive reasoning task**, functioning as a **Second Reader** that drafts clinically grounded reports for radiologist review rather than replacing human judgment.

---

## 🧠 Core Idea: Cognitive Simulation

Instead of a black-box encoder–decoder, CogRRG explicitly models **three stages of radiologist cognition**:

1. **Hierarchical Visual Perception (PRO-FA)**
2. **Diagnosis Formation (MIX-MLP)**
3. **Closed-Loop Hypothesis Verification (RCTA)**

This design improves **clinical efficiency**, **reduces hallucinations**, and provides **interpretable intermediate representations**.

---

## 🧩 Cognitive Modules

### 1️⃣ PRO-FA: Hierarchical Visual Perception

Radiologists examine CXRs at multiple granularities: global organs, regional lobes, and fine-grained lesions.  
PRO-FA mirrors this behavior using a **Swin-Base Transformer**.

- Extracts **pixel-level**, **region-level**, and **organ-level** tokens
- Processes multi-view CXRs (PA/AP/Lateral) with shared weights
- Aligns visual tokens with a curated **CXR-RadLex concept bank** using contrastive loss

This grounding ensures the model learns *what anatomical and pathological concepts look like*, rather than relying on spurious correlations.

---

### 2️⃣ MIX-MLP: Knowledge-Enhanced Diagnosis Formation

Before writing reports, radiologists form an internal diagnostic hypothesis.  
MIX-MLP explicitly models this step.

- Predicts **14 CheXpert pathologies** using a multi-label classifier
- Dual-path architecture:
  - **Residual Path** for stable linear separability
  - **Expansion Path** for modeling disease co-occurrence
- Trained with weighted BCE and label smoothing to handle noisy labels

The predicted pathology vector serves as a **diagnostic hypothesis** that conditions report generation.

---

### 3️⃣ RCTA: Triangular Cognitive Attention (Verification Loop)

RCTA implements a **closed-loop reasoning mechanism** inspired by clinical verification:

1. **Contextualization:** Image features attend to clinical indication text (e.g., age, sex, symptoms)
2. **Hypothesis Formation:** Context attends to predicted pathology embeddings
3. **Verification:** Hypotheses re-attend to fine-grained image tokens for evidence checking

This triangular attention loop significantly reduces hallucinations and enforces clinical consistency during generation.

---

## 📊 Datasets

### MIMIC-CXR (Primary Training Dataset)
- 377,110 images from 227,835 studies
- PA, AP, and Lateral views
- Free-text radiology reports
- Used for end-to-end training and evaluation

### IU X-Ray (Domain Generalization Benchmark)
- 7,470 images from 3,955 reports
- Explicit Findings and Impression sections
- Treated as an **out-of-distribution benchmark** to test robustness and generalization

---

## 📈 Evaluation Metrics

CogRRG is evaluated using **clinically grounded metrics**, not just text similarity:

| Category | Metric |
|--------|--------|
| Clinical Accuracy | CheXpert F1 |
| Structural Reasoning | RadGraph F1 |
| NLG Fluency | CIDEr, BLEU-4 |

### Key Results
- **CheXpert F1:** 0.56  
- **RadGraph F1:** 0.52  
- **CIDEr:** 0.45  

![WhatsApp Image 2026-02-09 at 4 49 02 AM](https://github.com/user-attachments/assets/7c05dd39-8fa4-458c-8983-9416c6489f3e)

All metrics exceed the hackathon benchmark thresholds.

---

## 🧩 Interpretability

CogRRG provides interpretability at multiple levels:
- **Concept Heatmaps:** Visual–RadLex alignment maps
- **Explicit Diagnosis Hypotheses:** Pathology probabilities before generation
- **Structural Verification:** RadGraph-based relation analysis
