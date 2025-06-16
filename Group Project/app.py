import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from sklearn.decomposition import TruncatedSVD
import numpy as np


# Load the dataset
dataframe = pd.read_csv("dataframe_with_cat.csv")



## APP CONFIGURATION ------------------------------------------------
# Inject CSS for customized styling
st.markdown(
    """
    <style>
    /* Gradient background for the entire app with updated colors */
    .stApp {
        background: linear-gradient(45deg, #800080, #DA70D6, #BA55D3, #8A2BE2);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Override Streamlit's container styling */
    .block-container {
        background-color: white;
        border-radius: 15px; /* Rounded corners */
        padding: 0px 20px 20px 20px;
        box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
        max-width: 900px;
        margin: 80px auto;
        max-height: calc(120vh - 40px);
        overflow-y: auto;
    }

    .custom-title {
        margin-top: 0px;
        margin-bottom: 5px;
    }
    .block-container h1:first-of-type {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }

    /* Style buttons with dynamic gradient hover effect */
    .stButton > button {
        background: linear-gradient(45deg, black, black);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        border: none;
        transition: background 0.5s ease-in-out, color 0.5s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #000000, #1a1a1a, #333333);
        color: white;
    }

    .stButton > button:active {
        background: black !important; /* Black effect on press */
        color: white !important;
    }

    @keyframes gradientHover {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* Center-align titles, labels, and text */
    .block-container h1, 
    .block-container h2, 
    .block-container h3, 
    .block-container h4, 
    .block-container h5, 
    .block-container h6,
    .block-container p, 
    .block-container label, 
    .block-container .stMarkdown {
        text-align: center !important;
    }

    /* Center-align Streamlit buttons */
    .stButton {
        display: flex;
        justify-content: center;
        margin: 0 auto 1em auto;
    }

    /* Center-align DataFrame outputs */
    .stDataFrame, .stTable {
        margin-left: auto !important;
        margin-right: auto !important;
        display: table !important;
    }
    
    /* Full-width tabs styling: distributing two tabs equally to occupy the container */
    [data-baseweb="tab-list"] {
         width: 100%;
         display: flex;
         justify-content: center;
    }
    [data-baseweb="tab"] {
         flex: 0 0 50%;
         text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Main container for layout consistency
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title for the marketplace with a shopping emoji and custom styling
st.markdown('<h1 class="custom-title">🛍️ Nova Market: Your Ultimate Marketplace</h1>', unsafe_allow_html=True)

# Create two tabs: New User and Existing User
tab_new, tab_existing = st.tabs(["New User", "Existing User"])


# ------------------ New User Container Tab------------------
with tab_new:
    st.markdown("### Welcome, new explorer!")

    # Create two columns for the selectboxes
    col1, col2 = st.columns(2)

    with col1:
        # Define Category options with "All"
        categories = sorted(dataframe["Category"].unique())
        category_search = st.selectbox("Looking for a specific Category", ["All"] + categories)

    with col2:
        # If a category is chosen, filter subcategories by that category 
        if category_search != "All":
            subcategories = sorted(dataframe[dataframe["Category"] == category_search]["Subcategory"].unique())
        else:
            subcategories = sorted(dataframe["Subcategory"].unique())
        subcategory_search = st.selectbox("What are you looking for? (Products)", ["All"] + subcategories)
    
    if st.button("Look for it"):
        df_filtered = dataframe.copy()
        if category_search != "All":
            df_filtered = df_filtered[df_filtered["Category"].str.startswith(category_search, na=False)]
        if subcategory_search != "All":
            df_filtered = df_filtered[df_filtered["Subcategory"].str.startswith(subcategory_search, na=False)]
        
        
        # Group by Subcategory and Description, then aggregate total Quantity and average UnitPrice
        df_grouped = df_filtered.groupby(["Subcategory", "Description"], as_index=False).agg({
            "Quantity": "sum",
            "UnitPrice": "mean"
        })
        
        # Get the top 6 products based on highest total Quantity
        df_top6 = df_grouped.sort_values(by="Quantity", ascending=False).head(6)

        n_products = len(df_top6)
        st.markdown(f"### Top {n_products} Products")

        # Loop through the products two at a time and create two columns per row.
        for i in range(0, len(df_top6), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                # Ensure we don't go out-of-bounds if the number isn't exactly even
                if i + j < len(df_top6):
                    row = df_top6.iloc[i + j]
                    # Set star rating based on the product's overall rank
                    if (i + j) == 0:
                        rating = "⭐" * 5
                    elif (i + j) in [1, 2]:
                        rating = "⭐" * 4
                    else:
                        rating = "⭐" * 3
                    
                    # Display each product in a styled container within the column
                    col.markdown(
                        f"""
                        <div style="
                            border: 2px solid #ddd; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 15px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            background-color: #f9f9f9;
                        ">
                            <h4 style="margin: 5px 0;">{row['Subcategory']}</h4>
                            <p style="margin: 5px 0;">{row['Description']}</p>
                            <p style="margin: 5px 0;">Unit Price: {row['UnitPrice']:.2f}</p>
                            <p style="margin: 5px 0;">Rating: {rating}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
    #  Top Last Purchases Container

    # Convert the "Date" column to datetime (if not already converted)
    dataframe["Date_dt"] = pd.to_datetime(dataframe["Date"])

    # Use the maximum date from the dataset to avoid issues if the dataset is from the past
    max_date = dataframe["Date_dt"].max()
    two_weeks_ago = max_date - pd.Timedelta(weeks=2)

    # Filter the dataset for the last two weeks
    df_recent = dataframe[dataframe["Date_dt"] >= two_weeks_ago]

    # Group the recent data by Category and Description to get the total Quantity sold
    df_recent_grouped = df_recent.groupby(["Category", "Description"], as_index=False).agg({
        "Quantity": "sum"
    })

    # Get the top 10 products (by Quantity) from the last two weeks
    df_top5_last = df_recent_grouped.sort_values(by="Quantity", ascending=False).head(10)


    # Build slider items dynamically from df_top5_last
    carousel_items_content = ""
    for idx, row in df_top5_last.iterrows():
        carousel_items_content += f"""
        <div class="slide">
        <h5 style="margin: 5px 0; text-align: center; font-size: 20px;">{row['Category']}</h5>
        <p style="margin: 5px 0; text-align: center;">{row['Description']}</p>
        <p style="margin: 5px 0; text-align: center;">Total Sold: {row['Quantity']} 🔥</p>
        </div>
        """
    # Duplicate slides for continuous effect.
    full_carousel_items = carousel_items_content + carousel_items_content

    # Calculate total width for the slide track (each slide is 250px wide)
    n = len(df_top5_last)
    slide_track_width = 250 * n * 2

    carousel_html = f"""
    <!-- Include Google Fonts for Source Sans Pro -->
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap" rel="stylesheet">

    <style>
    /* Slider container styling */
    .slider {{
    width: 100%;
    overflow: hidden;
    position: relative;
    border: 1px solid #ddd;
    margin-top: 20px;
    font-family: 'Source Sans Pro', sans-serif;
    
    }}

    /* The track holding all slides, with dynamic width */
    .slide-track {{
    display: flex;
    width: {slide_track_width}px;
    animation: scroll 50s linear infinite;
    font-family: 'Source Sans Pro', sans-serif;
    }}

    /* Updated slide styling using a prettier container effect */
    .slide {{
    flex: 0 0 250px; /* fixed width for each slide */
    margin: 5px;
    padding: 15px;
    background-color: #f9f9f9;
    border: 2px solid #ddd;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 150px;  /* fixed height to maintain uniform look */
    font-family: 'Source Sans Pro', sans-serif;
    }}

    /* Continuous scrolling animation */
    @keyframes scroll {{
    0% {{
        transform: translateX(0);
    }}
    100% {{
        transform: translateX(-50%);
    }}
    }}
    </style>

    <div class="slider">
    <div class="slide-track">
        {full_carousel_items}
    </div>
    </div>
    """

    st.markdown("###  What's on fire these weeks? 🔥 ")
    components.html(carousel_html, height=220)




# -------------------------------------------------------------------------------------


## SECOND TAB: Existing User


with tab_existing:
    st.markdown("<h3 style='text-align: center;'>Welcome back!</h3>", unsafe_allow_html=True)

    # Get sorted unique user IDs
    user_ids = sorted(dataframe["CustomerID"].unique())

    # Determine the default index for user ID 14995
    default_index = user_ids.index(14995) if 14995 in user_ids else 0

    # Create three columns and place the selectbox in the center
    left, center, right = st.columns(3)
    with center:
        selected_user = st.selectbox("Select your User ID", user_ids, index=default_index)

    # Filter data for the selected user
    user_data = dataframe[dataframe["CustomerID"] == selected_user]

    if not user_data.empty:
        total_purchases = user_data["Quantity"].sum()
        last_purchase_date = pd.to_datetime(user_data["Date"]).max().strftime("%Y-%m-%d")
        user_country = user_data["Country"].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Purchases", f"{total_purchases:,}")
        col2.metric("Last Purchase", last_purchase_date)
        col3.metric("Country", user_country)

        recent_items = user_data.sort_values(by="InvoiceDate", ascending=False).drop_duplicates(subset=["Description"])
        top_3_recent = recent_items.head(3)



        ## Matrix Model Creation
        # Create customer-item matrix (Quantity-based or binary)
        customer_item_matrix = dataframe.pivot_table(index="CustomerID", 
                                                    columns="Description", 
                                                    values="Quantity", 
                                                    aggfunc="sum", 
                                                    fill_value=0)


        # Factorize into k latent features
        n_components = 20  # tunable
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd.fit_transform(customer_item_matrix)
        reconstructed_matrix = pd.DataFrame(np.dot(latent_matrix, svd.components_),
                                            index=customer_item_matrix.index,
                                            columns=customer_item_matrix.columns)
        
        # Get predicted scores for selected user
        user_scores = reconstructed_matrix.loc[selected_user]

        # Exclude items the user has already purchased
        already_purchased = customer_item_matrix.loc[selected_user]
        unseen_mask = already_purchased == 0
        predicted_scores = user_scores[unseen_mask]

        # Convert to DataFrame and merge in product metadata
        predicted_df = predicted_scores.reset_index()
        predicted_df.columns = ["Description", "PredictedScore"]

        # Merge with original dataframe to recover Category and UnitPrice
        # Collapse duplicates by Description, averaging price and picking most common category/subcategory
        product_info = (
            dataframe
            .groupby("Description", as_index=False)
            .agg({
                "Category": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
                "Subcategory": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
                "UnitPrice": "mean"
            })
        )
        predicted_df = predicted_df.merge(product_info, on="Description", how="left")


        st.markdown("---")
        st.markdown("#### 🔎 Explore by Products")

        available_subcats = sorted(predicted_df["Subcategory"].dropna().unique())
        selected_subcat_table = st.selectbox("Choose a subcategory", available_subcats, key="subcat_table")
        

        subcat_filtered_df = predicted_df[predicted_df["Subcategory"] == selected_subcat_table]
        subcat_filtered_df = subcat_filtered_df.sort_values(by="PredictedScore", ascending=False)

        subcat_filtered_df["UnitPrice"] = subcat_filtered_df["UnitPrice"].round(2)
        # Normalize predicted scores to a 1–5 scale
        min_score = subcat_filtered_df["PredictedScore"].min()
        max_score = subcat_filtered_df["PredictedScore"].max()

        # Avoid division by zero in case all values are the same
        if max_score != min_score:
            subcat_filtered_df["StarScore"] = ((subcat_filtered_df["PredictedScore"] - min_score) / (max_score - min_score)) * 4 + 1
        else:
            subcat_filtered_df["StarScore"] = 3  # fallback to neutral

        # Round to nearest integer and convert to star emojis
        subcat_filtered_df["Likeliness"] = subcat_filtered_df["StarScore"].round().astype(int).apply(lambda x: "⭐" * x)

        display_table = subcat_filtered_df[["Description", "UnitPrice", "Likeliness"]].copy()
        display_table["Buy"] = "🛒 Buy"

        st.dataframe(display_table.reset_index(drop=True), use_container_width=True)



        st.markdown("---")
        
        # Subcategory dropdown
        category_options = sorted(predicted_df["Category"].dropna().unique())
        selected_subcat = st.selectbox("Search by Category", category_options)

        # Filter recommendations by selected subcategory
        filtered_df = predicted_df[predicted_df["Category"] == selected_subcat]
        top6 = filtered_df.sort_values(by="PredictedScore", ascending=False).head(6)
        n_products = len(top6)
        st.markdown(f"### Top {n_products} Recommended Products")

        # Display in 2 columns per row
        for i in range(0, len(top6), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(top6):
                    row = top6.iloc[i + j]

                    # Optional: Rating logic based on ranking
                    rating = "⭐" * (5 if i + j == 0 else 4 if i + j in [1, 2] else 3)

                    col.markdown(
                        f"""
                        <div style="
                            border: 2px solid #ddd;
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            background-color: #f9f9f9;
                            font-family: 'Source Sans Pro', sans-serif;
                        ">
                            <h4 style="margin: 5px 0;">{row['Subcategory']}</h4>
                            <p style="margin: 5px 0;">{row['Description']}</p>
                            <p style="margin: 5px 0;">Unit Price: {row['UnitPrice']:.2f}</p>
                            <p style="margin: 5px 0;">Rating: {rating}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        st.markdown("---")

        st.markdown("#### 🛍️ Buy again")
        cols = st.columns(3)
        for i, (idx, row) in enumerate(top_3_recent.iterrows()):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 15px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        background-color: #f9f9f9;
                        text-align: center;
                        font-family: 'Source Sans Pro', sans-serif;
                        min-height: 200px;
                        max-height: 200px;
                    ">
                        <h5 style="margin: 5px 0;">{row['Subcategory']}</h5>
                        <p style="margin: 5px 0;">{row['Description']}</p>
                        <p style="margin: 5px 0;">Unit Price: {row['UnitPrice']:.2f}</p>
                        <p style="margin: 5px 0; color: green; font-weight: bold;">Buy again</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )