import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# --- Streamlit Import ---
import streamlit as st

# --- 1. Data Loading and Preprocessing (Modified for Streamlit Caching) ---

@st.cache_data # This decorator caches the data loading and preprocessing results
def load_and_preprocess_data(movies_path, ratings_path, users_path):
    """
    Loads the datasets and performs initial cleaning and preprocessing.
    Cached by Streamlit to run only once.
    """
    print(f"Loading data from: {movies_path}, {ratings_path}, {users_path}") # This will print to the terminal where streamlit runs
    
    try:
        movies_df = pd.read_csv(movies_path, encoding='utf-8')
        ratings_df = pd.read_csv(ratings_path, encoding='utf-8')
        users_df = pd.read_csv(users_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Trying 'latin1' encoding for all files...")
        movies_df = pd.read_csv(movies_path, encoding='latin1')
        ratings_df = pd.read_csv(ratings_path, encoding='latin1')
        users_df = pd.read_csv(users_path, encoding='latin1')
    except Exception as e:
        st.error(f"An unexpected error occurred while loading files: {e}")
        st.stop() # Stop the app if data loading fails critically
        
    print("Data loaded successfully. Starting preprocessing...")

    # Preprocessing Movies DataFrame
    movies_df['Year'] = movies_df['Title'].str.extract(r'\((\d{4})\)')
    movies_df['Year'] = pd.to_numeric(movies_df['Year'])
    movies_df['Title'] = movies_df['Title'].str.replace(r' \(\d{4}\)', '', regex=True)
    movies_df['Category'] = movies_df['Category'].apply(lambda x: x.split('|'))
    movies_exploded_df = movies_df.explode('Category')
    print("Movie data preprocessed.")

    # Preprocessing Users DataFrame
    age_map = {
        1: 'Under 18', 18: '18-24', 25: '25-34', 35: '35-44',
        45: '45-49', 50: '50-55', 56: 'Above 56'
    }
    users_df['AgeGroup'] = users_df['Age'].map(age_map)

    occupation_map = {
        0: 'Not specified or other', 1: 'Academician', 2: 'Artist',
        3: 'Admin/Office work', 4: 'Grad/Higher Ed student', 5: 'Customer Service/Consultant',
        6: 'Doctor and Medical services', 7: 'Executive and Managerial',
        8: 'Farmer and Agriculture', 9: 'Homemaker', 10: 'K-12 Student',
        11: 'Lawyer', 12: 'Programmer', 13: 'Retired',
        14: 'Sales and Marketing', 15: 'Scientist', 16: 'Self-Employed',
        17: 'Engineer and Tradesman/Craftsman', 18: 'Technician',
        19: 'Unemployed', 20: 'Writer'
    }
    users_df['OccupationName'] = users_df['Occupation'].map(occupation_map)
    print("User data preprocessed.")

    print("All data loading and preprocessing complete.")
    return movies_df, movies_exploded_df, ratings_df, users_df # Return all needed DFs

# --- 2. Data Analysis and Mining Techniques (Queries i, ii, iii, v, vi) ---
# (Keep these functions as they are, they will be called by the Streamlit app)

def query_total_movies_per_year(movies_df):
    """Query i) Total number of movies released in each year."""
    movies_per_year = movies_df.groupby('Year').size().reset_index(name='Count')
    return movies_per_year.sort_values('Year')

def query_highest_rated_category_per_year(movies_exploded_df, ratings_df):
    """Query ii) Find the movie category having highest ratings in each year."""
    merged_df = pd.merge(movies_exploded_df, ratings_df, on='MovieID')
    avg_ratings = merged_df.groupby(['Year', 'Category'])['Rating'].mean().reset_index()
    idx = avg_ratings.groupby('Year')['Rating'].transform(max) == avg_ratings['Rating']
    highest_rated_category_per_year = avg_ratings[idx].sort_values(['Year', 'Category'])
    count_movies = merged_df.groupby(['Year', 'Category'])['MovieID'].nunique().reset_index(name='CountOfMovies')
    result = pd.merge(highest_rated_category_per_year, count_movies, on=['Year', 'Category'])
    return result[['Year', 'Category', 'CountOfMovies']]

def query_category_age_group_likings(movies_exploded_df, ratings_df, users_df):
    """Query iii) Find and display movie category and age group wise likings."""
    merged_user_ratings = pd.merge(ratings_df, users_df, on='UserID')
    final_merged_df = pd.merge(merged_user_ratings, movies_exploded_df, on='MovieID')
    likings = final_merged_df.groupby(['AgeGroup', 'Category'])['UserID'].nunique().reset_index(name='NumberOfUsers')
    idx = likings.groupby('AgeGroup')['NumberOfUsers'].transform(max) == likings['NumberOfUsers']
    most_liked_category_per_age_group = likings[idx].sort_values(['AgeGroup', 'NumberOfUsers'], ascending=[True, False])
    return most_liked_category_per_age_group

def query_year_wise_count_movies(movies_df):
    """Query v) Display year wise count of movies released."""
    return query_total_movies_per_year(movies_df)

def query_year_category_wise_count_movies(movies_exploded_df):
    """Query vi) Display year wise, category wise count of movies released."""
    year_category_counts = movies_exploded_df.groupby(['Year', 'Category']).size().reset_index(name='Count')
    return year_category_counts.sort_values(['Year', 'Category'])

# --- 3. Clustering Models (Query iv and Query vii Part 1) ---
# (Keep these functions as they are, they will be called by the Streamlit app)

def cluster_category_age_likings(movies_exploded_df, ratings_df, users_df, n_clusters=3):
    """Query iv) Use Cluster models to segregate movie category and age group wise likings."""
    merged_user_ratings = pd.merge(ratings_df, users_df, on='UserID')
    final_merged_df = pd.merge(merged_user_ratings, movies_exploded_df, on='MovieID')
    age_category_pivot = final_merged_df.groupby(['AgeGroup', 'Category'])['UserID'].nunique().unstack(fill_value=0)
    age_category_pivot = age_category_pivot.fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(age_category_pivot)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    age_category_pivot['Cluster'] = kmeans.fit_predict(scaled_features)
    return age_category_pivot # Return the pivot table with cluster assignments

def cluster_category_occupation_likings(movies_exploded_df, ratings_df, users_df, n_clusters=3):
    """Query vii) Part 1: Use Clustering methods to segregate movie category and occupation of users."""
    merged_user_ratings = pd.merge(ratings_df, users_df, on='UserID')
    final_merged_df = pd.merge(merged_user_ratings, movies_exploded_df, on='MovieID')
    occupation_category_pivot = final_merged_df.groupby(['OccupationName', 'Category'])['UserID'].nunique().unstack(fill_value=0)
    occupation_category_pivot = occupation_category_pivot.fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(occupation_category_pivot)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    occupation_category_pivot['Cluster'] = kmeans.fit_predict(scaled_features)
    return occupation_category_pivot # Return the pivot table with cluster assignments

# --- 4. Predictive Models (Query vii Part 2 - Occupation-based Liking Prediction) ---
# (Keep these functions as they are, they will be called by the Streamlit app)

def train_occupation_liking_model(movies_exploded_df, ratings_df, users_df):
    """Trains a predictive model to predict movie likings for a given occupation."""
    merged_data = pd.merge(ratings_df, users_df, on='UserID')
    merged_data = pd.merge(merged_data, movies_exploded_df, on='MovieID')
    occupation_category_avg_rating = merged_data.groupby(['OccupationName', 'Category'])['Rating'].mean().unstack(fill_value=0)
    X = pd.get_dummies(occupation_category_avg_rating.index).set_index(occupation_category_avg_rating.index)
    Y = occupation_category_avg_rating
    X = X.reindex(Y.index)
    categories = Y.columns.tolist()
    occupation_names_for_encoding = X.columns.tolist()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    st.write(f"Model Evaluation (Occupation-based): MSE={mse:.4f}, R2={r2:.4f}") # Display in Streamlit
    return model, categories, occupation_names_for_encoding

def predict_likings_for_occupation(model, categories, occupation_names_for_encoding, target_occupation_name):
    """Predicts movie likings for a given occupation."""
    input_vector = np.zeros(len(occupation_names_for_encoding))
    try:
        occ_index = occupation_names_for_encoding.index(target_occupation_name)
        input_vector[occ_index] = 1
    except ValueError:
        st.error(f"Error: Occupation '{target_occupation_name}' not found in the training data.")
        st.write("Please choose from the following occupations:", occupation_names_for_encoding)
        return None
    input_df = pd.DataFrame([input_vector], columns=occupation_names_for_encoding)
    predicted_likings_raw = model.predict(input_df)[0]
    predicted_likings = pd.Series(predicted_likings_raw, index=categories)
    return predicted_likings.sort_values(ascending=False)

# --- 5. Refine Predictive Model with Age Group (Query viii) ---
# (Keep these functions as they are, they will be called by the Streamlit app)

def train_refined_liking_model_with_age_occupation(movies_exploded_df, ratings_df, users_df):
    """Trains a refined predictive model by including AgeGroup in addition to OccupationName."""
    merged_data = pd.merge(ratings_df, users_df, on='UserID')
    merged_data = pd.merge(merged_data, movies_exploded_df, on='MovieID')
    age_occ_category_avg_rating = merged_data.groupby(['AgeGroup', 'OccupationName', 'Category'])['Rating'].mean().unstack(fill_value=0)
    X_raw = pd.DataFrame(age_occ_category_avg_rating.index.tolist(), columns=['AgeGroup', 'OccupationName'])
    X = pd.get_dummies(X_raw)
    Y = age_occ_category_avg_rating.reset_index(drop=True)
    categories = Y.columns.tolist()
    feature_columns = X.columns.tolist()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    st.write(f"Refined Model Evaluation (Age & Occupation): MSE={mse:.4f}, R2={r2:.4f}") # Display in Streamlit
    return model, categories, feature_columns

def predict_likings_for_age_occupation(model, categories, feature_columns, target_age_group, target_occupation_name):
    """Predicts movie likings for a given age group and occupation."""
    input_vector = np.zeros(len(feature_columns))
    age_col_name = f'AgeGroup_{target_age_group}'
    occ_col_name = f'OccupationName_{target_occupation_name}'
    age_found = False
    occ_found = False
    try:
        if age_col_name in feature_columns:
            input_vector[feature_columns.index(age_col_name)] = 1
            age_found = True
        else:
            st.error(f"Error: Age Group '{target_age_group}' not found in training data features.")
        if occ_col_name in feature_columns:
            input_vector[feature_columns.index(occ_col_name)] = 1
            occ_found = True
        else:
            st.error(f"Error: Occupation '{target_occupation_name}' not found in training data features.")
        if not age_found or not occ_found:
            return None
    except Exception as e:
        st.error(f"An error occurred while preparing input vector: {e}")
        return None
    input_df = pd.DataFrame([input_vector], columns=feature_columns)
    predicted_likings_raw = model.predict(input_df)[0]
    predicted_likings = pd.Series(predicted_likings_raw, index=categories)
    return predicted_likings.sort_values(ascending=False)

# --- 6. Predictive Model: Category to Age Group & Occupation (Query ix) ---
# (Keep these functions as they are, they will be called by the Streamlit app)

def train_category_demographic_model(movies_exploded_df, ratings_df, users_df):
    """Trains a predictive model to predict the most likely age group and occupation for a movie category."""
    merged_data = pd.merge(ratings_df, users_df, on='UserID')
    merged_data = pd.merge(merged_data, movies_exploded_df, on='MovieID')
    likings_data = merged_data.groupby(['Category', 'AgeGroup', 'OccupationName'])['UserID'].nunique().reset_index(name='NumberOfUsers')
    idx = likings_data.groupby('Category')['NumberOfUsers'].idxmax()
    most_liked_demographics_per_category = likings_data.loc[idx].reset_index(drop=True)
    X = pd.get_dummies(most_liked_demographics_per_category['Category'], prefix='Category')
    Y_age = most_liked_demographics_per_category['AgeGroup']
    Y_occ = most_liked_demographics_per_category['OccupationName']
    age_group_classes = sorted(Y_age.unique())
    occupation_classes = sorted(Y_occ.unique())
    category_columns = X.columns.tolist()
    Y = pd.DataFrame({
        'AgeGroup': Y_age,
        'OccupationName': Y_occ
    })
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=42)
    base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(base_classifier)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)
    age_accuracy = accuracy_score(Y_test['AgeGroup'], Y_pred_df['AgeGroup'])
    occ_accuracy = accuracy_score(Y_test['OccupationName'], Y_pred_df['OccupationName'])
    st.write(f"Category-Demographic Model Evaluation: Age Acc={age_accuracy:.4f}, Occ Acc={occ_accuracy:.4f}") # Display in Streamlit
    return model, age_group_classes, occupation_classes, category_columns

def predict_demographics_for_category(model, age_group_classes, occupation_classes, category_columns, target_category):
    """Predicts the most likely age group and occupation for a given movie category."""
    input_vector = np.zeros(len(category_columns))
    category_col_name = f'Category_{target_category}'
    if category_col_name not in category_columns:
        st.error(f"Error: Category '{target_category}' not found in the training data features.")
        st.write("Please choose from the following categories:", [col.replace('Category_', '') for col in category_columns])
        return None
    input_vector[category_columns.index(category_col_name)] = 1
    input_df = pd.DataFrame([input_vector], columns=category_columns)
    predicted_demographics_raw = model.predict(input_df)
    predicted_age_group = predicted_demographics_raw[0][0]
    predicted_occupation_name = predicted_demographics_raw[0][1]
    return predicted_age_group, predicted_occupation_name


# --- Streamlit App Layout (Step 7) ---

def main():
    st.set_page_config(layout="wide") # Use wide layout for better display
    st.title("🎬 Movie Data Mining and Analytics System")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", 
                            ["Home", 
                             "Data Analysis Queries", 
                             "Clustering Models", 
                             "Predictive Models",
                             "Category to Demographics Model"])

    # Define paths to your CSV files (adjust if they are in a different folder)
    movies_path = "Movies.csv"
    ratings_path = "Ratings.csv"
    users_path = "Users.csv"

    # Load and preprocess data once using caching
    with st.spinner('Loading and preprocessing data...'):
        movies_df, movies_exploded_df, ratings_df, users_df = load_and_preprocess_data(
            movies_path, ratings_path, users_path
        )
    st.sidebar.success("Data Loaded!")

    if page == "Home":
        st.header("Welcome to the Movie Analysis Dashboard!")
        st.write("""
            This application allows you to explore movie data, understand audience likings,
            and leverage predictive models based on user demographics.
            Use the navigation on the left to explore different functionalities.
        """)
        st.subheader("Sample Data Overview:")
        st.write("Movies Data (first 5 rows):")
        st.dataframe(movies_df.head())
        st.write("Ratings Data (first 5 rows):")
        st.dataframe(ratings_df.head())
        st.write("Users Data (first 5 rows):")
        st.dataframe(users_df.head())

    elif page == "Data Analysis Queries":
        st.header("📊 Data Analysis Queries")
        st.write("Here you can run basic queries to get insights from the movie data.")

        st.subheader("1. Total Movies Released Per Year")
        if st.button("Run Query 1"):
            result = query_total_movies_per_year(movies_df)
            st.dataframe(result)
        
        st.subheader("2. Highest Rated Movie Category Per Year")
        if st.button("Run Query 2"):
            result = query_highest_rated_category_per_year(movies_exploded_df, ratings_df)
            st.dataframe(result)

        st.subheader("3. Movie Category and Age Group Wise Likings")
        st.info("Based on the highest number of unique users from each age group rating a category.")
        if st.button("Run Query 3"):
            result = query_category_age_group_likings(movies_exploded_df, ratings_df, users_df)
            st.dataframe(result)

        st.subheader("4. Year-wise Count of Movies Released (Same as Query 1)")
        if st.button("Run Query 4"):
            result = query_year_wise_count_movies(movies_df)
            st.dataframe(result)

        st.subheader("5. Year-wise, Category-wise Count of Movies Released")
        if st.button("Run Query 5"):
            result = query_year_category_wise_count_movies(movies_exploded_df)
            st.dataframe(result)

    elif page == "Clustering Models":
        st.header("🧩 Clustering Models")
        st.write("These models group similar age groups or occupations based on movie category preferences.")

        st.subheader("1. Movie Category and Age Group Likings Clustering (Query iv)")
        n_clusters_age = st.slider("Select number of clusters for Age Group Likings:", min_value=2, max_value=7, value=3)
        if st.button("Run Age Group Clustering"):
            with st.spinner(f"Clustering Age Groups into {n_clusters_age} clusters..."):
                age_group_clusters = cluster_category_age_likings(movies_exploded_df, ratings_df, users_df, n_clusters=n_clusters_age)
                st.dataframe(age_group_clusters[['Cluster']])
                st.subheader("Average 'liking' (unique users) per Category within each Cluster:")
                cluster_analysis = age_group_clusters.groupby('Cluster').mean()
                for cluster_id in sorted(age_group_clusters['Cluster'].unique()):
                    st.write(f"**--- Cluster {cluster_id} ---**")
                    st.dataframe(cluster_analysis.loc[cluster_id].sort_values(ascending=False).head(5))

        st.subheader("2. Movie Category and Occupation Likings Clustering (Query vii Part 1)")
        n_clusters_occ = st.slider("Select number of clusters for Occupation Likings:", min_value=2, max_value=7, value=3)
        if st.button("Run Occupation Clustering"):
            with st.spinner(f"Clustering Occupations into {n_clusters_occ} clusters..."):
                occupation_clusters = cluster_category_occupation_likings(movies_exploded_df, ratings_df, users_df, n_clusters=n_clusters_occ)
                st.dataframe(occupation_clusters[['Cluster']])
                st.subheader("Average 'liking' (unique users) per Category within each Cluster:")
                cluster_analysis_occ = occupation_clusters.groupby('Cluster').mean()
                for cluster_id in sorted(occupation_clusters['Cluster'].unique()):
                    st.write(f"**--- Cluster {cluster_id} ---**")
                    st.dataframe(cluster_analysis_occ.loc[cluster_id].sort_values(ascending=False).head(5))

    elif page == "Predictive Models":
        st.header("🔮 Predictive Models")
        st.write("These models predict movie likings based on user demographics.")

        # Cache the trained models so they don't retrain on every interaction
        @st.cache_resource
        def get_occupation_liking_model(movies_exp_df, r_df, u_df):
            st.info("Training Occupation-based Liking Model (this runs once)...")
            return train_occupation_liking_model(movies_exp_df, r_df, u_df)

        @st.cache_resource
        def get_refined_liking_model(movies_exp_df, r_df, u_df):
            st.info("Training Refined Liking Model (this runs once)...")
            return train_refined_liking_model_with_age_occupation(movies_exp_df, r_df, u_df)

        # Train models (cached)
        trained_model_q7, predicted_categories_q7, all_occupations_for_encoding_q7 = get_occupation_liking_model(movies_exploded_df, ratings_df, users_df)
        trained_model_q8, predicted_categories_q8, all_features_q8 = get_refined_liking_model(movies_exploded_df, ratings_df, users_df)

        st.subheader("1. Predict Movie Likings by Occupation (Query vii Part 2)")
        available_occupations_q7 = sorted(users_df['OccupationName'].dropna().unique().tolist())
        selected_occupation_q7 = st.selectbox("Select an Occupation:", available_occupations_q7, key='occ_q7_select')
        if st.button("Predict Likings for Selected Occupation"):
            with st.spinner(f"Predicting for {selected_occupation_q7}..."):
                predicted_likings = predict_likings_for_occupation(
                    trained_model_q7, predicted_categories_q7, all_occupations_for_encoding_q7, selected_occupation_q7
                )
                if predicted_likings is not None:
                    st.write(f"**Top 5 Liked Categories for '{selected_occupation_q7}':**")
                    st.dataframe(predicted_likings.head(5))

        st.subheader("2. Predict Movie Likings by Age Group and Occupation (Query viii)")
        available_age_groups_q8 = sorted(users_df['AgeGroup'].dropna().unique().tolist())
        available_occupations_q8 = sorted(users_df['OccupationName'].dropna().unique().tolist())
        
        selected_age_group_q8 = st.selectbox("Select an Age Group:", available_age_groups_q8, key='age_q8_select')
        selected_occupation_q8 = st.selectbox("Select an Occupation:", available_occupations_q8, key='occ_q8_select')
        
        if st.button("Predict Likings for Age & Occupation"):
            with st.spinner(f"Predicting for {selected_age_group_q8} and {selected_occupation_q8}..."):
                predicted_likings_refined = predict_likings_for_age_occupation(
                    trained_model_q8, predicted_categories_q8, all_features_q8, 
                    selected_age_group_q8, selected_occupation_q8
                )
                if predicted_likings_refined is not None:
                    st.write(f"**Top 5 Liked Categories for Age Group '{selected_age_group_q8}' and Occupation '{selected_occupation_q8}':**")
                    st.dataframe(predicted_likings_refined.head(5))

    elif page == "Category to Demographics Model":
        st.header("🎯 Category to Demographics Model")
        st.write("This model predicts the most likely age group and occupation for a given movie category.")

        @st.cache_resource
        def get_category_demographic_model(movies_exp_df, r_df, u_df):
            st.info("Training Category-Demographic Model (this runs once)...")
            return train_category_demographic_model(movies_exp_df, r_df, u_df)
        
        trained_model_q9, age_classes_q9, occ_classes_q9, all_category_features_q9 = get_category_demographic_model(movies_exploded_df, ratings_df, users_df)

        available_categories_q9 = sorted(movies_exploded_df['Category'].dropna().unique().tolist())
        selected_category_q9 = st.selectbox("Select a Movie Category:", available_categories_q9, key='cat_q9_select')

        if st.button("Predict Demographics for Category"):
            with st.spinner(f"Predicting demographics for '{selected_category_q9}'..."):
                predicted_demographics = predict_demographics_for_category(
                    trained_model_q9, age_classes_q9, occ_classes_q9, all_category_features_q9, selected_category_q9
                )
                if predicted_demographics is not None:
                    st.write(f"**Predicted Demographics for Category '{selected_category_q9}':**")
                    st.write(f"**Most Likely Age Group:** {predicted_demographics[0]}")
                    st.write(f"**Most Likely Occupation:** {predicted_demographics[1]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()