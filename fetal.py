# App to predict the class of fetal health using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Create title
st.title('Fetal Health Classification: A Machine Learning App')

# Display the gif
st.image('fetal_health_image.gif', width = 600)

# Create first subheader
st.subheader("Utilize our advanced Machine Learning application to predict fetal health classifications.")

# Create text for user guidance
st.write("To ensure optimal results, please ensure that your data strictly adheres to the specified format outlined below:")
# Load fetal_health.csv file into a DataFrame
fetal_df = pd.read_csv('fetal_health.csv')
# Drop fetal_health column because that will not be present in user-provided data
fetal_df = fetal_df.drop(columns = ['fetal_health'])
# Display the DataFrame head in Streamlit, with index hidden
st.dataframe(fetal_df.head(), hide_index=True)

# Asking users to upload their CSV file
user_file = st.file_uploader("Upload your data")

# Create second subheader
st.subheader("Predicting Fetal Health Class")

# Reading the pickle file that were created in fetal.ipynb 
rf_pickle = open('rf_fetal.pickle', 'rb') 
clf = pickle.load(rf_pickle) 
rf_pickle.close() 

# Trigger following once user uploads their CSV file
if user_file is not None:
    # Loading data
    user_df = pd.read_csv(user_file) # User provided data
    original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model
    # Remove output (fetal_health) column from original data
    original_df = original_df.drop(columns = ['fetal_health'])

    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Split data into original and user dataframes using row index
    original_df_split = combined_df[:original_rows]
    user_df_split = combined_df[original_rows:]

    # Predictions and probabilities for user data
    user_pred = clf.predict(user_df_split.values)
    user_probs = clf.predict_proba(user_df_split.values)

    # Create new columns in user_df for predictions and probabilities
    user_df['Predicted Fetal Health'] = user_pred
    user_df['Prediction Probability (%)'] = (user_probs.max(axis=1) * 100).round(1)

    # Define a function to apply conditional styling
    def color_cells(val):
        if val == 'Normal':
            return 'background-color: lime'
        elif val == 'Suspect':
            return 'background-color: yellow'
        elif val == 'Pathological':
            return 'background-color: orange'
        else:
            return ''

    # Apply the styling function to the 'Predicted Fetal Health' column
    styled_user_df = user_df.style.applymap(color_cells, subset=['Predicted Fetal Health'])

    # Display the styled DataFrame in Streamlit
    st.write(styled_user_df)

# Create third subheader
st.subheader("Prediction Performance")
# Showing additional items
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
with tab1:
    st.image('feature_importance.svg')
with tab2:
    st.image('confusion_matrix.svg')
with tab3:
    df = pd.read_csv('classification_report.csv', index_col = 0)
    st.dataframe(df)