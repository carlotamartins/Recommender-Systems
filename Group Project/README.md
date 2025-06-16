
# 🛒 Recommender System for Online Retail (Final Project)

This project implements a hybrid recommender system for a UK-based online retail business that sells a wide range of products (bazaar-style). It includes a machine learning pipeline and an interactive **Streamlit app** for real-time product recommendations.

---

## 📁 Project Structure

```
Recommender Systems/
├── Group Project/
│   ├── Code/
│   │   ├── app.py                       # Streamlit app interface
│   │   ├── dataframe_with_cat.xlsx      # Product data with category info
│   │   ├── Group Project.ipynb          # Full modeling pipeline and evaluation
│   │   ├── OnlineRetail.xlsx            # Raw transactional dataset
│   │   └── outputs/                     # Serialized model artifacts
│   │       ├── cosine_df.pkl
│   │       ├── customer_item_matrix.pkl
│   │       ├── reconstructed_df.pkl
│   │       ├── svd_predicted_matrix.pkl
│   │       ├── train_matrix.pkl
│   │       └── user_based_predicted_cosine.pkl
```

---

## 🗂️ Setup Instructions

### Step 1: Unzip the Dataset

- Download and unzip the dataset ZIP file provided separately.
- Place the unzipped contents (e.g., `OnlineRetail.xlsx`, `dataframe_with_cat.xlsx`) into the `Code/` folder.

### Step 2: Run the Jupyter Notebook

Before using the app, generate the model outputs:

```bash
# Inside Code/
jupyter notebook Group\ Project.ipynb
```

Run all cells to generate recommendation matrices and save model files into the `outputs/` folder.

### Step 3: Launch the Streamlit App

Once the notebook has finished running:

```bash
cd Code
streamlit run app.py
```

---


## 📊 Data Sources

- `OnlineRetail.xlsx`: Cleaned transactional dataset.
- 

## 🙌 Authors

Group Project – Recommender Systems  
ESADE Business School  
June 2025

---

