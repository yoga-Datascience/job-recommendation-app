# feedback.py

import pandas as pd
import streamlit as st
import os
import base64
import json
import requests

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
                f"How relevant is '{row['Job Title']}' to your job search?",
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

def save_feedback(feedback, github_token_env_var='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', repo_name='yoga-Datascience/job-recommendation-app', file_path='feedback.txt', commit_message='Update feedback', branch='main'):
    if feedback is not None and len(feedback) > 0:
        try:
            # Get the GitHub token from environment variable
            github_token = os.getenv(github_token_env_var)
            if not github_token:
                st.error("GitHub token not found. Please set it as an environment variable.")
                return False

            # GitHub API URL for the file
            api_url = f'https://api.github.com/repos/{repo_name}/blob/main/{file_path}'

            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Get the current file content and SHA
            response = requests.get(api_url, headers=headers, params={'ref': branch})
            if response.status_code == 200:
                file_info = response.json()
                sha = file_info['sha']
                existing_content = base64.b64decode(file_info['content']).decode('utf-8')
            elif response.status_code == 404:
                # File does not exist, so no SHA or existing content
                sha = None
                existing_content = ''
            else:
                st.error(f"Failed to get file info from GitHub: {response.json().get('message', 'Unknown error')}")
                return False

            # Convert feedback to text
            new_feedback_entries = []
            for entry in feedback:
                new_feedback_entries.append(json.dumps(entry))
            new_content = existing_content + '\n' + '\n'.join(new_feedback_entries)

            # Encode content to Base64
            encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')

            # Prepare data for the PUT request
            data = {
                'message': commit_message,
                'content': encoded_content,
                'branch': branch
            }
            if sha:
                data['sha'] = sha

            # Update the file
            response = requests.put(api_url, headers=headers, data=json.dumps(data))

            if response.status_code in [200, 201]:
                return True
            else:
                st.error(f"Failed to update file on GitHub: {response.json().get('message', 'Unknown error')}")
                return False

        except Exception as e:
            st.error(f"An error occurred while saving feedback: {e}")
            return False
    else:
        # No feedback to save
        return False
