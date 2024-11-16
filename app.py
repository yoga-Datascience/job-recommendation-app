import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Set page configuration
st.set_page_config(page_title="Job Recommendation System", layout="wide")

st.title("Job Recommendation System")

# User Inputs
st.header("Step 1: User Inputs")

# Salary Input
salary = st.number_input("Expected Salary (in USD)", min_value=0)

# Prestige Slider
prestige = st.slider("Prestige Level (1-5)", 1, 5, step=1)

# Caching data loading functions
@st.cache_data
def load_geography_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_job_data(file_path):
    return pd.read_csv(file_path, delimiter="\t")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_bert_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Load geography data
try:
    geography_df = load_geography_data("Geography.csv")
    st.success("Geography Data Loaded Successfully!")
except FileNotFoundError:
    st.error("Geography data file not found. Please ensure 'Geography.csv' is in the same directory.")

# Location Selection
try:
    states = geography_df["State"].unique()
    selected_state = st.selectbox("Select State", options=[""] + list(states))

    # Dynamically load CountyTownName based on the selected state
    if selected_state:
        counties = geography_df[geography_df["State"] == selected_state]["CountyTownName"].unique()
        selected_county = st.selectbox("Select County/Town", options=[""] + list(counties))
    else:
        selected_county = None
except Exception as e:
    st.error(f"Error loading location data: {e}")

# Education Qualification
education = st.selectbox("Education Qualification", ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Ph.D."])

# Job Description
job_description = st.text_area("Describe Your Desired Job", height=200)

# Display Inputs on Submit
if st.button("Submit"):
    st.subheader("Inputs Received:")
    st.write(f"**Expected Salary:** ${salary}")
    st.write(f"**Prestige Level:** {prestige}")
    st.write(f"**Location:** {selected_state}, {selected_county}")
    st.write(f"**Education:** {education}")
    st.write(f"**Job Description:** {job_description}")

# Load models
nlp = load_spacy_model()
model = load_bert_model()

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

# Preprocess job description
if job_description:
    processed_description = preprocess_text(job_description)
    st.markdown(f"**Preprocessed Job Description:** {processed_description}")

# Load job duties data
try:
    duties_df = load_job_data("All_duties.txt")
    st.success("Dataset Loaded Successfully!")
    st.dataframe(duties_df.head())
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'All_duties.txt' is in the same directory.")

# Initialize TF-IDF Vectorizer
@st.cache_resource
def get_tfidf_matrix(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=2, ngram_range=(1, 2))
    return vectorizer.fit_transform(data)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=2, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(duties_df["Job_Duties"].fillna(""))

# Encode job duties using BERT
job_embeddings = model.encode(duties_df["Job_Duties"].fillna("").tolist(), convert_to_tensor=True)

def find_top_matches(user_input, tfidf_vectorizer, tfidf_matrix, model, job_embeddings, top_n=5):
    # Preprocess user input
    user_input_processed = preprocess_text(user_input)

    # TF-IDF similarity
    tfidf_vector = tfidf_vectorizer.transform([user_input_processed])
    tfidf_scores = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()

    # BERT similarity
    user_embedding = model.encode(user_input_processed, convert_to_tensor=True)
    bert_scores = util.pytorch_cos_sim(user_embedding, job_embeddings)[0].cpu().numpy()

    # Combine scores
    combined_scores = 0.5 * tfidf_scores + 0.5 * bert_scores

    # Get top matches
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    return duties_df.iloc[top_indices], combined_scores[top_indices]

if st.button("Find Matches"):
    if job_description:
        top_matches, scores = find_top_matches(job_description, tfidf_vectorizer, tfidf_matrix, model, job_embeddings)
        st.session_state["top_matches"] = top_matches
        st.session_state["scores"] = scores

        st.subheader("Top Matches:")
        for i, (index, row) in enumerate(top_matches.iterrows()):
            st.markdown(f"### **Match {i + 1}: {row['Occupation']}**")
            st.write(row["Job_Duties"])
            st.write(f"**Score:** {scores[i]:.2f}")
    else:
        st.error("Please enter a job description to find matches.")

def filter_recommendations(df, salary, prestige):
    return df[(df["Salary"] >= salary) & (df["Prestige"] <= prestige)]

if st.button("Filter Recommendations"):
    if "top_matches" in st.session_state:
        top_matches = st.session_state["top_matches"]
        filtered_matches = filter_recommendations(top_matches, salary, prestige)
        st.subheader("Filtered Matches:")
        st.dataframe(filtered_matches)
    else:
        st.error("No matches found. Please run 'Find Matches' first.")

