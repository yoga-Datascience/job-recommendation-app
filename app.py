import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from feedback import collect_feedback, save_feedback

# ----------------------------- #
#       Data Loading Section     #
# ----------------------------- #

@st.cache_data(show_spinner=False)
def load_data():
    """Load all necessary CSV data files."""
    salary_df = pd.read_csv('Salary.csv', low_memory=False)
    duties_df = pd.read_csv('Duties.csv')
    education_df = pd.read_csv('Education.csv')
    geography_df = pd.read_csv('Geography.csv')
    prestige_df = pd.read_csv('Prestige.csv')
    return salary_df, duties_df, education_df, geography_df, prestige_df

salary_df, duties_df, education_df, geography_df, prestige_df = load_data()

# ----------------------------- #
#       Data Preprocessing       #
# ----------------------------- #

def standardize_soc_code(soc_code):
    """Standardize the SOC Code by removing decimals and non-digit characters."""
    soc_code = str(soc_code).strip()
    soc_code = soc_code.split('.')[0]  # Remove decimal part
    soc_code = ''.join(filter(lambda x: x.isdigit() or x == '-', soc_code))
    return soc_code

# Apply standardization to relevant DataFrames
for df in [salary_df, duties_df, education_df, prestige_df]:
    df['Soc Code'] = df['Soc Code'].apply(standardize_soc_code)
    df['Soc Code'] = df['Soc Code'].astype(str).str.strip().astype('category')

# Merge salary_df with geography_df to get 'State' and 'County' information
salary_df = salary_df.merge(
    geography_df[['Area', 'State', 'CountyTownName']],
    on='Area',
    how='left'
)
salary_df['State'] = salary_df['State'].astype('category')
salary_df['CountyTownName'] = salary_df['CountyTownName'].astype('category')

# Preprocess education_df
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
education_df['Soc Code'] = education_df['Soc Code'].astype('category')

# Preprocess prestige_df
prestige_df['GSS Ratings 2012'] = prestige_df['GSS Ratings 2012'].fillna(0)
prestige_min = prestige_df['GSS Ratings 2012'].min()
prestige_max = prestige_df['GSS Ratings 2012'].max()
prestige_df['Prestige_Normalized'] = 1 + 4 * (prestige_df['GSS Ratings 2012'] - prestige_min) / (prestige_max - prestige_min)
prestige_df['Prestige_Normalized'] = prestige_df['Prestige_Normalized'].round(2)
prestige_df['Soc Code'] = prestige_df['Soc Code'].astype('category')

# Preprocess duties_df
duties_df['Job_Duties'] = duties_df['Job_Duties'].fillna('')
duties_df['Soc Code'] = duties_df['Soc Code'].astype('category')

# Extract job duties list
job_duties_list = duties_df['Job_Duties'].tolist()

# ----------------------------- #
#       Model & Embeddings       #
# ----------------------------- #

@st.cache_resource(show_spinner=True)
def load_model():
    """Load the SentenceTransformer model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

@st.cache_data(show_spinner=True)
def compute_job_embeddings(_model, job_duties):
    """Compute and cache job embeddings as NumPy arrays."""
    job_embeddings = _model.encode(job_duties, convert_to_tensor=False, show_progress_bar=True)
    return job_embeddings

job_embeddings = compute_job_embeddings(model, job_duties_list)

# ----------------------------- #
#           UI Section           #
# ----------------------------- #

st.title("Job Recommendation System")

# Salary expectation input
salary_input = st.number_input(
    "Please enter your salary expectation:",
    min_value=0,
    max_value=int(salary_df['Average'].max()),
    step=5000,
    value=int(salary_df['Average'].median())
)

# Education level input
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

# Prestige level input
prestige_input = st.slider(
    "Please enter your expected level of prestige (1 to 5):",
    min_value=1,
    max_value=5,
    step=1,
    value=3
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

# ----------------------------- #
#         Recommendation         #
# ----------------------------- #

if st.button("Find Jobs"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        try:
            # ----------------------------- #
            #   Step 1: Filter DataFrames    #
            # ----------------------------- #

            # 1.1 Filter salary_df by salary_input, state_input, and county_input
            filtered_salary = salary_df[
                (salary_df['Average'] >= salary_input) &
                (salary_df['State'] == state_input) &
                (salary_df['CountyTownName'] == county_input)
            ].copy()
            filtered_salary = filtered_salary.drop_duplicates(subset='Soc Code')

            # 1.2 Filter education_df by education_input
            filtered_education = education_df[education_df['Education_Rank'] <= education_input].copy()
            filtered_education = filtered_education.drop_duplicates(subset='Soc Code')

            # 1.3 Filter prestige_df by prestige_input
            filtered_prestige = prestige_df[prestige_df['Prestige_Normalized'] >= prestige_input].copy()
            filtered_prestige = filtered_prestige.drop_duplicates(subset='Soc Code')

            # ----------------------------- #
            #   Step 2: Compute Similarity    #
            # ----------------------------- #

            # Encode the user's job description
            user_embedding = model.encode(job_description, convert_to_tensor=False)

            # Compute cosine similarities using scikit-learn
            cosine_similarities = cosine_similarity([user_embedding], job_embeddings)[0]

            # Create a local copy of duties_df and add similarity scores
            duties_df_local = duties_df.copy()
            duties_df_local['Similarity'] = cosine_similarities

            # Find the top N matching job titles
            top_n = 5  # Adjust as needed
            top_matches = duties_df_local.sort_values(by='Similarity', ascending=False).head(top_n)
            top_matches = top_matches.drop_duplicates(subset='Soc Code')

            # ----------------------------- #
            #   Step 3: Merge Filters        #
            # ----------------------------- #

            # Get the intersection of SOC Codes across all filters
            soc_codes = set(filtered_salary['Soc Code']) & set(filtered_education['Soc Code']) & set(filtered_prestige['Soc Code']) & set(top_matches['Soc Code'])

            # Filter the top_matches DataFrame based on the intersected SOC Codes
            final_df = top_matches[top_matches['Soc Code'].isin(soc_codes)].copy()

            if not final_df.empty:
                # ----------------------------- #
                #   Step 4: Enrich Final Data    #
                # ----------------------------- #

                # Add 'Average Salary', 'State', and 'County' from filtered_salary
                salary_mapping = filtered_salary.set_index('Soc Code')[['Average', 'State', 'CountyTownName']]
                final_df = final_df.merge(salary_mapping, on='Soc Code', how='left')

                # Add 'Minimum Education Qualification' from filtered_education
                education_mapping = filtered_education.set_index('Soc Code')['Education']
                final_df = final_df.merge(education_mapping, on='Soc Code', how='left')

                # Add 'Prestige Score' from filtered_prestige
                prestige_mapping = filtered_prestige.set_index('Soc Code')['Prestige_Normalized']
                final_df = final_df.merge(prestige_mapping, on='Soc Code', how='left')

                # Rename and reorder columns
                final_df = final_df.rename(columns={
                    'Soc Code': 'SOC Code',
                    'Occupation': 'Job Title',
                    'Average': 'Average Salary',
                    'CountyTownName': 'County',
                    'Education': 'Minimum Education Qualification',
                    'Prestige_Normalized': 'Prestige Score',
                    'Similarity': 'Similarity Score'
                })

                final_df['Similarity Score'] = final_df['Similarity Score'].round(4)

                # Select and order columns
                final_df = final_df[[
                    'SOC Code',
                    'Job Title',
                    'State',
                    'County',
                    'Average Salary',
                    'Minimum Education Qualification',
                    'Prestige Score',
                    'Similarity Score'
                ]]

                # Sort by 'Similarity Score' descending
                final_df = final_df.sort_values(by='Similarity Score', ascending=False)

                # ----------------------------- #
                #     Step 5: Display Results     #
                # ----------------------------- #

                # Function to create O*NET link
                def create_onet_link(soc_code):
                    formatted_soc_code = ''.join(filter(str.isdigit, str(soc_code)))
                    return f"https://www.onetonline.org/link/summary/{formatted_soc_code}.00"

                # Add hyperlinks to 'Job Title'
                final_df['Job Title'] = final_df.apply(
                    lambda row: f"<a href='{create_onet_link(row['SOC Code'])}' target='_blank'>{row['Job Title']}</a>",
                    axis=1
                )

                st.markdown("### Recommended Jobs for You:")
                st.markdown(final_df.to_html(escape=False, index=False), unsafe_allow_html=True)

                # ----------------------------- #
                #       Step 6: Collect Feedback #
                # ----------------------------- #

                feedback, submit_feedback = collect_feedback(
                    final_df,
                    salary_input,
                    education_input_label,
                    prestige_input,
                    state_input,
                    county_input,
                    job_description
                )

                if submit_feedback:
                    if feedback is not None:
                        success = save_feedback(feedback)
                        if success:
                            st.success("Thank you for your feedback!")
                        else:
                            st.error("Feedback not saved!")
                    else:
                        st.error("No feedback collected.")
            else:
                st.warning("No jobs found matching your criteria.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ----------------------------- #
#       Additional Notes         #
# ----------------------------- #

st.markdown("""
---
*Note: Ensure that all CSV files (`Salary.csv`, `Duties.csv`, `Education.csv`, `Geography.csv`, `Prestige.csv`) are correctly placed in the application's directory. Additionally, the `feedback` module should contain the `collect_feedback` and `save_feedback` functions as expected.*
""")
