# feedback.py

import pandas as pd
import streamlit as st
import os

def collect_feedback(final_df, salary_input, education_input_label, prestige_input, state_input, county_input, job_description):
    st.markdown("### Please rate the relevance of the recommended jobs:")
    rating_to_numeric = {
        'Not Rated': None,
        'Not Relevant': 1.0,
        'Somewhat Relevant': 2.0,
        'Relevant': 3.0,
        'Highly Relevant': 4.0
    }

    with st.form(key='feedback_form'):
        # Collect rating inputs
        ratings = {}
        for index, row in final_df.iterrows():
            rating = st.selectbox(
                f"How relevant is '{row['SOC Code']}' to your job search?",
                ['Not Rated', 'Not Relevant', 'Somewhat Relevant', 'Relevant', 'Highly Relevant'],
                index=0,
                key=f"rating_{index}_{row['SOC Code']}"
            )
            ratings[index] = rating
        submit_feedback = st.form_submit_button("Submit Feedback")

    feedback = []

    if submit_feedback:
        for index, row in final_df.iterrows():
            rating = ratings[index]
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
    else:
        feedback = None  # Feedback is None if form is not submitted

    # Return both feedback and the submit_feedback flag
    return feedback, submit_feedback

def save_feedback(feedback, filename='feedback.csv'):
    if feedback is not None and len(feedback) > 0:
        feedback_df = pd.DataFrame(feedback)
        # Ensure consistent column order
        columns = ['SOC Code', 'Job Title', 'User Rating', 'Salary Expectation', 'Education Level',
                   'Prestige Level', 'State', 'County', 'Job Description']
        feedback_df = feedback_df[columns]

        try:
            # Check if the file exists
            if os.path.exists(filename):
                # Append to the existing file without writing the header
                feedback_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                # Write to a new file, including the header
                feedback_df.to_csv(filename, index=False)
            return True
        except Exception as e:
            st.error(f"An error occurred while saving feedback: {e}")
            return False
    else:
        # No feedback to save
        return False
