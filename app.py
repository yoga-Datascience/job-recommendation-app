import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the data files
salary_df = pd.read_csv('/home/yoga/Archive/Projects/job-recommendation-app/Salary.csv')
duties_df = pd.read_csv('/home/yoga/Archive/Projects/job-recommendation-app/Duties.csv')
education_df = pd.read_csv('/home/yoga/Archive/Projects/job-recommendation-app/Education.csv')
geography_df = pd.read_csv('/home/yoga/Archive/Projects/job-recommendation-app/Geography.csv')
prestige_df = pd.read_csv('/home/yoga/Archive/Projects/job-recommendation-app/Prestige.csv')

# Standardize 'Soc Code's across DataFrames
def standardize_soc_code(soc_code):
    soc_code = str(soc_code).strip()
    soc_code = soc_code.split('.')[0]  # Remove decimal part
    soc_code = ''.join(filter(lambda x: x.isdigit() or x == '-', soc_code))
    return soc_code

# Apply the cleaning function to all DataFrames
for df in [salary_df, duties_df, education_df, prestige_df]:
    df['Soc Code'] = df['Soc Code'].apply(standardize_soc_code)
    df['Soc Code'] = df['Soc Code'].astype(str).str.strip()

# Merge salary_df with geography_df to get 'State' and 'County' information
salary_df = salary_df.merge(geography_df[['Area', 'State', 'CountyTownName']], on='Area', how='left')

# Load BERT model and encode job duties once
@st.cache(allow_output_mutation=True)
def load_model_and_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    duties_df['Job_Duties'] = duties_df['Job_Duties'].fillna('')
    job_duties_list = duties_df['Job_Duties'].tolist()
    job_embeddings = model.encode(job_duties_list, convert_to_tensor=True)
    return model, job_embeddings

model, job_embeddings = load_model_and_embeddings()

# User Inputs
st.title("Job Recommendation System")

# Salary expectation
salary_input = st.number_input(
    "Please enter your salary expectation:",
    min_value=0,
    max_value=int(salary_df['Average'].max()),
    step=5000
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
    sorted(geography_df['State'].unique())
)

# County input based on selected state
county_options = geography_df[geography_df['State'] == state_input]['CountyTownName'].unique()
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
        # Step 1: Filter DataFrames Based on User Inputs

        # 1.1 Filter salary_df by salary_input, state_input, and county_input
        filtered_salary = salary_df[
            (salary_df['Average'] >= salary_input) &
            (salary_df['State'] == state_input) &
            (salary_df['CountyTownName'] == county_input)
        ]
        filtered_salary = filtered_salary.drop_duplicates(subset='Soc Code')

        # 1.2 Filter education_df by education_input
        education_level_mapping = {
            'none': 0,
            'high school diploma or equivalent': 1,
            "associate's degree": 2,
            "bachelor's degree": 3,
            "master's degree": 4,
            'doctoral or professional degree': 5
        }
        education_df['Education'] = education_df['Education'].str.lower().str.strip()
        education_df['Education_Rank'] = education_df['Education'].map(education_level_mapping)
        education_df['Education_Rank'] = education_df['Education_Rank'].fillna(0).astype(int)
        # Include jobs that require the user's education level or lower
        filtered_education = education_df[education_df['Education_Rank'] <= education_input]
        filtered_education = filtered_education.drop_duplicates(subset='Soc Code')

        # 1.3 Filter prestige_df by prestige_input
        prestige_df['GSS Ratings 2012'] = prestige_df['GSS Ratings 2012'].fillna(0)
        prestige_min = prestige_df['GSS Ratings 2012'].min()
        prestige_max = prestige_df['GSS Ratings 2012'].max()
        prestige_df['Prestige_Normalized'] = 1 + 4 * (prestige_df['GSS Ratings 2012'] - prestige_min) / (prestige_max - prestige_min)
        filtered_prestige = prestige_df[prestige_df['Prestige_Normalized'] >= prestige_input]
        filtered_prestige = filtered_prestige.drop_duplicates(subset='Soc Code')

        # Step 2: Compute Similarities Using BERT
        # Encode the user's job description
        user_embedding = model.encode(job_description, convert_to_tensor=True)

        # Compute cosine similarities
        cosine_similarities = util.cos_sim(user_embedding, job_embeddings)[0]

        # Add similarities to the duties_df
        duties_df['Similarity'] = cosine_similarities.cpu().numpy()

        # Find the top N matching job titles
        top_n = 50
        top_matches = duties_df.sort_values(by='Similarity', ascending=False).head(top_n)
        top_matches = top_matches.drop_duplicates(subset='Soc Code')

        # Step 3: Perform VLOOKUP-like Filtering

        # Get the list of 'Soc Code's that meet all criteria
        soc_codes = set(filtered_salary['Soc Code']) & set(filtered_education['Soc Code']) & set(filtered_prestige['Soc Code']) & set(top_matches['Soc Code'])

        # Filter the top_matches DataFrame based on the 'soc_codes' set
        final_df = top_matches[top_matches['Soc Code'].isin(soc_codes)].copy()

        # Add 'Average Salary' and 'State', 'County' from filtered_salary using map()
        salary_mapping = filtered_salary.set_index('Soc Code')['Average']
        state_mapping = filtered_salary.set_index('Soc Code')['State']
        county_mapping = filtered_salary.set_index('Soc Code')['CountyTownName']

        final_df['Average Salary'] = final_df['Soc Code'].map(salary_mapping)
        final_df['State'] = final_df['Soc Code'].map(state_mapping)
        final_df['County'] = final_df['Soc Code'].map(county_mapping)

        # Add 'Minimum Education Qualification' from filtered_education using map()
        education_mapping = filtered_education.set_index('Soc Code')['Education']
        final_df['Minimum Education Qualification'] = final_df['Soc Code'].map(education_mapping)

        # Add 'Prestige Score' from filtered_prestige using map()
        prestige_mapping = filtered_prestige.set_index('Soc Code')['Prestige_Normalized']
        final_df['Prestige Score'] = final_df['Soc Code'].map(prestige_mapping)
        final_df['Prestige Score'] = final_df['Prestige Score'].round(2)

        # Rename columns
        final_df = final_df.rename(columns={
            'Soc Code': 'SOC Code',
            'Occupation': 'Job Title',
            'Similarity': 'Similarity Score'
        })

        # Select and reorder columns
        final_df = final_df[[
            'SOC Code', 'Job Title', 'State', 'County', 'Average Salary', 'Minimum Education Qualification', 'Prestige Score', 'Similarity Score'
        ]]

        # Sort the final DataFrame by 'Similarity Score' descending
        final_df = final_df.sort_values(by='Similarity Score', ascending=False)

        # Display the final DataFrame
        if not final_df.empty:
            # Add hyperlink to 'Job Title'
            def create_onet_link(soc_code):
                # Remove any non-digit characters to format the SOC code
                formatted_soc_code = ''.join(filter(str.isdigit, str(soc_code)))
                return f"https://www.onetonline.org/link/summary/{(soc_code+".00")}"

            final_df['Job Title'] = final_df.apply(
                lambda row: f"<a href='{create_onet_link(row['SOC Code'])}' target='_blank'>{row['Job Title']}</a>",
                axis=1
            )

            st.markdown("### Recommended Jobs for You:")
            st.markdown(final_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.warning("No jobs found matching your criteria.")