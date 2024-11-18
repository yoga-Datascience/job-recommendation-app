import pandas as pd
import streamlit as st
import os

def collect_feedback(final_df, salary_input, education_input_label, prestige_input, state_input, county_input, job_description):
    st.markdown("### Please rate the relevance of the recommended jobs:")
    feedback = []

    rating_to_numeric = {
        'Not Rated': None,
        'Not Relevant': 1.0,
        'Somewhat Relevant': 2.0,
        'Relevant': 3.0,
        'Highly Relevant': 4.0
    }

    with st.form(key='feedback_form'):
        for index, row in final_df.iterrows():
            rating = st.selectbox(
                f"How relevant is '{row['Job Title']}' to your job search?",
                ['Not Rated', 'Not Relevant', 'Somewhat Relevant', 'Relevant', 'Highly Relevant'],
                index=0,
                key=f"rating_{index}_{row['SOC Code']}"
            )
            feedback.append({
                'SOC Code': row['SOC Code'],
                'Job Title': row['Job Title'],
                'User Rating': rating_to_numeric[rating],
                'Salary Expectation': salary_input,
                'Education Level': education_input_label,
                'Prestige Level': prestige_input,
                'State': state_input,
                'County': county_input,
                'Job Description': job_description
            })
        submit_feedback = st.form_submit_button("Submit Feedback")

    # Return both feedback and the submit_feedback flag
    return feedback, submit_feedback

def save_feedback(feedback, filename='/home/yoga/Archive/Projects/job-recommendation-app/feedback.csv'):
    if feedback is not None:
        feedback_df = pd.DataFrame(feedback)
        # Ensure consistent column order
        columns = ['SOC Code', 'Job Title', 'User Rating', 'Salary Expectation', 'Education Level',
                   'Prestige Level', 'State', 'County', 'Job Description']
        feedback_df = feedback_df[columns]

        try:
            # Overwrite the file regardless of whether it exists
            feedback_df.to_csv(filename, index=False)
            return True
        except Exception as e:
            st.error(f"An error occurred while saving feedback: {e}")
            return False
    else:
        # No feedback to save
        return False
