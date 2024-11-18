import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from feedback import collect_feedback, save_feedback

# Optimize caching for data loading
@st.cache_data(show_spinner=False)
def load_data():
    # Load the data files
    salary_df = pd.read_csv('Salary.csv', low_memory=False)
    duties_df = pd.read_csv('Duties.csv')
    education_df = pd.read_csv('Education.csv')
    geography_df = pd.read_csv('Geography.csv')
    prestige_df = pd.read_csv('Prestige.csv')
    return salary_df, duties_df, education_df, geography_df, prestige_df

salary_df, duties_df, education_df, geography_df, prestige_df = load_data()

# Standardize 'Soc Code's across DataFrames
def standardize_soc_code(soc_code):
    soc_code = str(soc_code).strip()
    soc_code = soc_code.split('.')[0]  # Remove decimal part
    soc_code = ''.join(filter(lambda x: x.isdigit() or x == '-', soc_code))
    return soc_code

# Apply the cleaning function to all relevant DataFrames
for df in [salary_df, duties_df, education_df, prestige_df]:
    df['Soc Code'] = df['Soc Code'].apply(standardize_soc_code)
    df['Soc Code'] = df['Soc Code'].astype(str).str.strip()

# Merge salary_df with geography_df to get 'State' and 'County' information
salary_df = salary_df.merge(geography_df[['Area', 'State', 'CountyTownName']], on='Area', how='left')

# Optimize caching for model and embeddings
@st.cache_resource(show_spinner=False, max_entries=1)
def load_model_and_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    duties_df_clean = duties_df.copy()
    duties_df_clean['Job_Duties'] = duties_df_clean['Job_Duties'].fillna('')
    job_duties_list = duties_df_clean['Job_Duties'].tolist()
    job_embeddings = model.encode(job_duties_list, convert_to_tensor=True, show_progress_bar=True)
    return model, job_embeddings, duties_df_clean

model, job_embeddings, duties_df_clean = load_model_and_embeddings()

# Precompute mappings to speed up lookups later
salary_mapping = salary_df.set_index('Soc Code')[['Average', 'State', 'CountyTownName']].to_dict('index')
education_level_mapping = {
    'none': 0,
    'high school diploma or equivalent': 1,
    "associate's degree": 2,
    "bachelor's degree": 3,
    "master's degree": 4,
    'doctoral or professional degree': 5
}
education_df['Education'] = education_df['Education'].str.lower().str.strip()
education_df['Education_Rank'] = education_df['Education'].map(education_level_mapping).fillna(0).astype(int)
education_mapping = education_df.set_index('Soc Code')['Education'].to_dict()
prestige_min = prestige_df['GSS Ratings 2012'].min()
prestige_max = prestige_df['GSS Ratings 2012'].max()
prestige_df['Prestige_Normalized'] = 1 + 4 * (prestige_df['GSS Ratings 2012'] - prestige_min) / (prestige_max - prestige_min)
prestige_mapping = prestige_df.set_index('Soc Code')['Prestige_Normalized'].to_dict()

# Function to get top N similar jobs using efficient torch operations
def get_top_similar_jobs(user_embedding, job_embeddings, top_n=5):
    cosine_similarities = util.cos_sim(user_embedding, job_embeddings)[0]
    top_results = torch.topk(cosine_similarities, k=top_n)
    top_indices = top_results.indices.cpu().numpy()
    top_scores = top_results.values.cpu().numpy()
    return top_indices, top_scores

# Function to create O*NET links
def create_onet_link(soc_code):
    formatted_soc_code = ''.join(filter(str.isdigit, str(soc_code)))
    return f"https://www.onetonline.org/link/summary/{formatted_soc_code}.00"

# User Inputs
st.title("Job Recommendation System")

# Salary expectation
salary_input = st.number_input(
    "Please enter your salary expectation:",
    min_value=0,
    max_value=int(salary_df['Average'].max()),
    step=5
)

# Education level (ranked 0 to 5)
education_levels = {
    'None': 0,
    'High school diploma or equivalent': 1,
    "Associate's degree": 2,
    "Bachelor's degree": 3,
    "Master's degree": 4,
    'Doctoral or professional degree': 5
}
education_input_label = st.selectbox(
    "Please select your highest educational qualification:",
    list(education_levels.keys())
)
education_input = education_levels[education_input_label]

# Prestige level (scale of 1 to 5)
prestige_input = st.slider(
    "Please enter your expected level of prestige (1 to 5):",
    min_value=1,
    max_value=5,
    step=1
)

# State input
state_input = st.selectbox(
    "Please select the state you are interested in:",
    sorted(geography_df['State'].dropna().unique())
)

# County input based on selected state
county_options = geography_df[geography_df['State'] == state_input]['CountyTownName'].dropna().unique()
county_input = st.selectbox(
    "Please select the county you are interested in:",
    sorted(county_options)
)

# Job description input
job_description = st.text_area(
    "Describe the type of job you're looking for:",
    height=200
)

if st.button("Find Jobs"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        try:
            # Step 1: Filter Soc Codes Based on User Inputs

            # 1.1 Filter salary_df by salary_input, state_input, and county_input
            filtered_salary_soc = {
                soc for soc, data in salary_mapping.items()
                if data['Average'] >= salary_input and
                   data['State'] == state_input and
                   data['CountyTownName'] == county_input
            }

            # 1.2 Filter education_df by education_input
            filtered_education_soc = {
                soc for soc, rank in education_df.set_index('Soc Code')['Education_Rank'].items()
                if rank <= education_input
            }

            # 1.3 Filter prestige_df by prestige_input
            filtered_prestige_soc = {
                soc for soc, score in prestige_mapping.items()
                if score >= prestige_input
            }

            # Intersection of all filters
            eligible_soc_codes = filtered_salary_soc & filtered_education_soc & filtered_prestige_soc

            if not eligible_soc_codes:
                st.warning("No jobs found matching your criteria.")
            else:
                # Step 2: Compute Similarities Using BERT

                # Encode the user's job description
                user_embedding = model.encode(job_description, convert_to_tensor=True)

                # Get top N similar jobs
                top_n = 10  # Increase to get more candidates before filtering
                top_indices, top_scores = get_top_similar_jobs(user_embedding, job_embeddings, top_n)

                top_matches = duties_df_clean.iloc[top_indices].copy()
                top_matches['Similarity'] = top_scores

                # Filter top_matches by eligible_soc_codes
                top_matches = top_matches[top_matches['Soc Code'].isin(eligible_soc_codes)]

                # If top_matches are less than desired, consider increasing top_n
                if top_matches.empty:
                    st.warning("No jobs found matching your criteria after similarity filtering.")
                else:
                    # Add additional information from precomputed mappings
                    top_matches['Average Salary'] = top_matches['Soc Code'].map(
                        lambda soc: salary_mapping[soc]['Average'] if soc in salary_mapping else None
                    )
                    top_matches['State'] = top_matches['Soc Code'].map(
                        lambda soc: salary_mapping[soc]['State'] if soc in salary_mapping else None
                    )
                    top_matches['County'] = top_matches['Soc Code'].map(
                        lambda soc: salary_mapping[soc]['CountyTownName'] if soc in salary_mapping else None
                    )
                    top_matches['Minimum Education Qualification'] = top_matches['Soc Code'].map(
                        education_mapping
                    )
                    top_matches['Prestige Score'] = top_matches['Soc Code'].map(
                        prestige_mapping
                    ).round(2)

                    # Rename columns for clarity
                    top_matches = top_matches.rename(columns={
                        'Soc Code': 'SOC Code',
                        'Occupation': 'Job Title',
                        'Similarity': 'Similarity Score'
                    })

                    # Select and reorder columns
                    final_df = top_matches[[
                        'SOC Code', 'Job Title', 'State', 'County', 'Average Salary',
                        'Minimum Education Qualification', 'Prestige Score', 'Similarity Score'
                    ]]

                    # Sort the final DataFrame by 'Similarity Score' descending
                    final_df = final_df.sort_values(by='Similarity Score', ascending=False)

                    # Add hyperlink to 'Job Title'
                    final_df['Job Title'] = final_df['SOC Code'].apply(create_onet_link) \
                        .combine(final_df['Job Title'], lambda link, title: f"<a href='{link}' target='_blank'>{title}</a>")

                    st.markdown("### Recommended Jobs for You:")
                    st.markdown(final_df.to_html(escape=False, index=False), unsafe_allow_html=True)

                    # Collect user feedback using functions from feedback.py
                    # Ensure these functions are properly imported and handle errors
                    try:
                        feedback, submit_feedback = collect_feedback(
                            final_df,
                            salary_input,
                            education_input_label,
                            prestige_input,
                            state_input,
                            county_input,
                            job_description
                        )

                        # Only attempt to save feedback if the form was submitted
                        if submit_feedback:
                            if feedback is not None:
                                success = save_feedback(feedback)
                                if success:
                                    st.success("Thank you for your feedback!")
                                else:
                                    st.error("Feedback not saved!")
                            else:
                                st.error("No feedback collected.")
                    except Exception as e:
                        st.error(f"An error occurred while collecting feedback: {e}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
