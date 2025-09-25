
# Food Waste Optimization

This repository contains a machine learning project aimed at analyzing and predicting food wastage trends. By leveraging exploratory data analysis (EDA) and predictive modeling, the project seeks to optimize food usage and reduce wastage.

## Directory Structures 

```
prathapprr-food-waste-optimization/
├── EDA.ipynb                  # Jupyter notebook for Exploratory Data Analysis
├── app.py                     # Streamlit web application for user interaction
├── food_wastage_data.csv      # Dataset used for the project
├── food_wastage_model.joblib  # Trained machine learning model
├── model.ipynb                # Jupyter notebook for building and training the model
└── .ipynb_checkpoints/        # Auto-generated checkpoints for notebooks
    ├── Untitled-checkpoint.ipynb
    └── Untitled1-checkpoint.ipynb
```

## Project Overview

Food waste is a significant global issue, and this project focuses on using data science to identify patterns in food wastage and build predictive models to mitigate it. The workflow includes:
- **Exploratory Data Analysis (EDA):** To uncover patterns, correlations, and trends in the dataset.
- **Model Development:** Using machine learning to predict potential food wastage scenarios.
- **Web Application:** A user-friendly interface for visualizing insights and making predictions.

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`, `streamlit`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/prathapprr/food-waste-optimization.git
   cd prathapprr-food-waste-optimization
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Files Description

1. **`EDA.ipynb`:** Contains analysis of the dataset to extract useful insights and visualize food wastage trends.
2. **`model.ipynb`:** Includes the machine learning pipeline for data preprocessing, model training, and evaluation.
3. **`app.py`:** Implements a Streamlit web app for real-time user interaction, predictions, and visualizations.
4. **`food_wastage_data.csv`:** The dataset used for training and analysis.
5. **`food_wastage_model.joblib`:** A serialized version of the trained model for deployment.

## Usage

1. Launch the Streamlit application using the command:
   ```bash
   streamlit run app.py
   ```
2. Upload your data file (if applicable) through the web app interface.
3. Explore the visualized trends and make predictions using the trained model.

## Contribution

Contributions are welcome! If you have ideas to improve the project or fix any issues, feel free to:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or suggestions, feel free to reach out:
- **Developer:** Prathap
- **Email:** prathapy150@gmail.com
- **Location:** Bengaluru, Karnataka

---

Happy coding! 😊
