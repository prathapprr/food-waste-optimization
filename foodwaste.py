import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests  # For making API requests to Gemini

# Add custom background color styling using CSS for a professional color scheme
page_bg_color = '''
<style>
body {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5870 100%);
    color: #ffffff;
    font-family: "Arial", sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5870 100%);
    padding: 20px;
    border-radius: 10px;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
    text-align: center;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
}

.stButton button {
    background-color: #20b2aa;
    color: #ffffff;
    border-radius: 5px;
}

.stButton button:hover {
    background-color: #178d7e;
}

.sidebar .sidebar-content {
    background-color: #f0f4f8;
    color: #333333;
    border-radius: 10px;
    padding: 15px;
}

footer {
    color: #20b2aa;
    text-align: center;
}
</style>
'''

# Apply the CSS styles
st.markdown(page_bg_color, unsafe_allow_html=True)

# Set the title of the app with emoji and professional look
st.title("🍽️ Food Wastage Optimization in Restaurants")

# Introductory message with professional formatting
st.markdown("""
### 🥡 Predict & Optimize Your Restaurant's Food Waste!
This app helps you forecast potential food wastage based on various parameters in your restaurant. Save food, reduce costs, and optimize your operations.
""")

# Sidebar with app information and professional color scheme
st.sidebar.header("About")
st.sidebar.info("""
This tool helps restaurant managers predict food wastage based on input parameters like event type, location, and other features.
""")

# Path to the model file
model_path = 'food_wastage_model.pkl'

# Check if the model file exists
if not os.path.isfile(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the dataset to understand the feature set
    data = pd.read_csv('food_wastage_data.csv')

    # Dataset preview section
    st.subheader("👀 Dataset Preview")
    st.dataframe(data.head())

    # Statistics Section
    st.subheader("📊 Dataset Overview")
    st.write(data.describe())

    # Feature Input Section
    st.subheader("🔧 Input Parameters")

    # Initialize user input dictionary
    user_input_dict = {}

    # Organize features into two columns for better layout
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(data.columns):
        if data[feature].dtype == 'object':
            # Add categorical inputs as dropdown in one of the columns
            if i % 2 == 0:
                user_input_dict[feature] = col1.selectbox(feature, data[feature].unique())
            else:
                user_input_dict[feature] = col2.selectbox(feature, data[feature].unique())
        else:
            # Add numerical inputs in text input field without range limits
            if i % 2 == 0:
                user_input_dict[feature] = col1.text_input(feature)
            else:
                user_input_dict[feature] = col2.text_input(feature)

    # Prediction Button
    if st.button("🔮 Predict Food Waste"):
        try:
            # Convert user inputs to correct data types
            for feature in data.columns:
                if data[feature].dtype != 'object':
                    user_input_dict[feature] = float(user_input_dict[feature])

            # Convert the input dictionary to a DataFrame
            input_data = pd.DataFrame([user_input_dict])

            # One-hot encode the input data to match the training set
            input_data_encoded = pd.get_dummies(input_data)
            input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

            # Make predictions using the model
            prediction = model.predict(input_data_encoded)

            # Display the result in a success message
            st.success(f"Estimated Food Wastage: {prediction[0]:.2f} units 🍛")

            # Feature importance plot
            st.subheader("🔍 Feature Importance Visualization")
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': model.feature_names_in_, 'Importance': feature_importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='Blues_d')
            plt.title('Feature Importance for Food Waste Prediction')
            st.pyplot(plt)

        except ValueError as e:
            st.error(f"Input error: {e}")

# Footer with contact or links
st.sidebar.header("Need Help?")
st.sidebar.info("""
If you have any questions or issues with the prediction tool, feel free to contact us!
""")

st.sidebar.markdown("""
---
📧 Contact: [email@example.com](mailto:email@example.com)
🌐 Website: [www.restaurantoptimizer.com](http://www.restaurantoptimizer.com)
""")

# Interactive Data Exploration Section
st.subheader("📈 Interactive Data Exploration")

# Dropdown to select a specific feature for filtering
selected_feature = st.selectbox("Select a Feature to Filter:", data.columns)

# Slider for filtering numerical data based on selected feature
if data[selected_feature].dtype != 'object':
    min_value, max_value = float(data[selected_feature].min()), float(data[selected_feature].max())
    selected_value = st.slider(f"Filter {selected_feature}", min_value, max_value, (min_value, max_value))

    # Filtered dataset based on slider values
    filtered_data = data[(data[selected_feature] >= selected_value[0]) & (data[selected_feature] <= selected_value[1])]
else:
    # Filter by unique category if categorical
    selected_value = st.multiselect(f"Filter {selected_feature}", data[selected_feature].unique())
    filtered_data = data[data[selected_feature].isin(selected_value)]

# Display the filtered data dynamically
st.write(f"Showing data filtered by {selected_feature}:")
st.dataframe(filtered_data)

# Interactive chart using Altair
import altair as alt
chart = alt.Chart(filtered_data).mark_bar().encode(
    x=alt.X('Type of Food', title='Food Type'),
    y=alt.Y('Quantity of Food', title='Quantity'),
    color='Event Type'
).interactive()

st.altair_chart(chart)

# Download predictions as a CSV or Excel file
import io

st.subheader("📂 Download Your Predictions")

if 'prediction' in locals():
    # Create a separate container for download options
    with st.container():
        # Convert the dataframe to CSV
        csv = filtered_data.to_csv(index=False)

        # Button for downloading CSV
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='food_wastage_predictions.csv',
            mime='text/csv',
        )

        # Convert the dataframe to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_data.to_excel(writer, index=False, sheet_name='Predictions')
            writer.save()

        # Button for downloading Excel
        st.download_button(
            label="Download Predictions as Excel",
            data=output.getvalue(),
            file_name='food_wastage_predictions.xlsx',
            mime='application/vnd.ms-excel',
        )

# --------------------------------------
# Theme toggle (Dark/Light Mode)
st.sidebar.subheader("🎨 Change Theme")
theme = st.sidebar.radio(
    "Choose Theme", ['Light Mode', 'Dark Mode'], index=1
)

if theme == 'Dark Mode':
    page_bg_color = '''
    <style>
    body {
        background: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: #0e1117;
    }
    </style>
    '''
else:
    page_bg_color = '''
    <style>
    body {
        background: #ffffff;
        color: #000000;
    }
    .stApp {
        background: #ffffff;
    }
    </style>
    '''

# Apply the CSS styles for theme toggle
st.markdown(page_bg_color, unsafe_allow_html=True)

# --------------------------------------
# Chatbot with Gemini API
st.subheader("🤖 Ask Our AI Assistant")

# Direct assignment of the Gemini API key (Method 3)
GEMINI_API_KEY = "api key"

question = st.text_input("Ask a question about the tool or food wastage:")

if question:
    response = requests.post(
        "https://api.gemini.google.com/v1/completions",  # Example endpoint, check Gemini docs
        headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
        json={
            "prompt": question,
            "max_tokens": 150,
        }
    )
    if response.status_code == 200:
        assistant_answer = response.json()['choices'][0]['text']
        st.write(f"Assistant: {assistant_answer}")
    else:
        st.error(f"Error: {response.text}")
