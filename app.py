import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load('food_wastage_model.joblib')

# App title
st.title("Food Wastage Prediction and Data Analysis App")

# Tabs for navigation
tab1, tab2 = st.tabs(["Food Wastage Prediction", "Data Analysis Dashboard"])

# Tab 1: Food Wastage Prediction
with tab1:
    st.header("Enter the details of the event:")
    type_of_food = st.selectbox("Type of Food", ["Meat", "Vegetables", "Dairy", "Grains"])
    number_of_guests = st.number_input("Number of Guests", min_value=1)
    event_type = st.selectbox("Event Type", ["Corporate", "Birthday", "Wedding", "Other"])
    quantity_of_food = st.number_input("Quantity of Food (kg)", min_value=1)
    storage_conditions = st.selectbox("Storage Conditions", ["Refrigerated", "Room Temperature"])
    purchase_history = st.selectbox("Purchase History", ["Regular", "Occasional"])
    seasonality = st.selectbox("Seasonality", ["All Seasons", "Winter", "Summer", "Spring", "Autumn"])
    preparation_method = st.selectbox("Preparation Method", ["Buffet", "Plated", "Finger Food"])
    geographical_location = st.selectbox("Geographical Location", ["Urban", "Suburban", "Rural"])
    pricing = st.selectbox("Pricing", ["Low", "Moderate", "High"])

    if st.button("Predict Wastage Amount"):
        input_data = pd.DataFrame({
            "Type of Food": [type_of_food],
            "Number of Guests": [number_of_guests],
            "Event Type": [event_type],
            "Quantity of Food": [quantity_of_food],
            "Storage Conditions": [storage_conditions],
            "Purchase History": [purchase_history],
            "Seasonality": [seasonality],
            "Preparation Method": [preparation_method],
            "Geographical Location": [geographical_location],
            "Pricing": [pricing]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        st.success(f"Predicted Food Wastage Amount: {prediction[0]:.2f} kg")

# Tab 2: Data Analysis Dashboard
with tab2:
    st.header("Data Analysis Dashboard")

    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if uploaded_file:
        try:
            # Load the CSV file
            df = pd.read_csv(uploaded_file)

            # Display dataset information
            st.subheader("Dataset Overview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head())

            # Identify numerical and categorical columns
            numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

            # Sidebar: Multi-column filter
            st.sidebar.subheader("Filter Data")
            filters = {}
            for col in categorical_columns:
                unique_values = df[col].dropna().unique().tolist()
                selected_value = st.sidebar.multiselect(f"Filter {col}", unique_values)
                if selected_value:
                    filters[col] = selected_value

            # Apply filters
            filtered_df = df.copy()
            for col, values in filters.items():
                filtered_df = filtered_df[filtered_df[col].isin(values)]

            st.subheader("Filtered Data")
            st.write(filtered_df)

            # Download filtered data
            if st.button("Download Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )

            # Data summary
            st.subheader("Summary Statistics")
            if numerical_columns:
                st.write(filtered_df[numerical_columns].describe())
            else:
                st.write("No numerical columns available for summary statistics.")

            # Correlation heatmap
            if len(numerical_columns) > 1:
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    filtered_df[numerical_columns].corr(),
                    annot=True, cmap="coolwarm", fmt=".2f", ax=ax
                )
                st.pyplot(fig)

            # Visualization options
            st.subheader("Visualizations")

            # Boxplot
            if st.checkbox("Show Boxplot"):
                col_for_boxplot = st.selectbox("Select Column for Boxplot", numerical_columns)
                if col_for_boxplot:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=filtered_df, y=col_for_boxplot, ax=ax)
                    ax.set_title(f"Boxplot of {col_for_boxplot}")
                    st.pyplot(fig)

            # Histogram
            if st.checkbox("Show Histogram"):
                col_for_histogram = st.selectbox("Select Column for Histogram", numerical_columns)
                if col_for_histogram:
                    bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
                    fig, ax = plt.subplots()
                    ax.hist(filtered_df[col_for_histogram].dropna(), bins=bins, color="blue", alpha=0.7)
                    ax.set_title(f"Histogram of {col_for_histogram}")
                    ax.set_xlabel(col_for_histogram)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

            # Scatter plot
            if len(numerical_columns) > 1 and st.checkbox("Show Scatter Plot"):
                x_col = st.selectbox("X-axis", numerical_columns)
                y_col = st.selectbox("Y-axis", numerical_columns)
                if x_col and y_col:
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                    st.pyplot(fig)

            # Line chart
            if st.checkbox("Show Line Chart"):
                line_chart_col = st.selectbox("Select Column for Line Chart", numerical_columns)
                if line_chart_col:
                    st.line_chart(filtered_df[line_chart_col])

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a CSV file to get started.")
