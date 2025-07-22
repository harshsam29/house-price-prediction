House Price Prediction Project
Overview
This project builds a machine learning model to predict house prices using the Kaggle House Prices: Advanced Regression Techniques dataset. The model uses a Random Forest Regressor with hyperparameter tuning via GridSearchCV to improve prediction accuracy. The project includes feature engineering (e.g., TotalSF, HouseAge, RemodAge) and outlier removal. A Streamlit web app allows users to input house features and get price predictions. The app is deployable locally or on Streamlit Cloud.
Features

Dataset: Kaggle House Prices dataset (train.csv).
Model: Random Forest Regressor with tuned hyperparameters (n_estimators, max_depth, min_samples_split).
Preprocessing: Handles missing values, scales numerical features, and one-hot encodes categorical features.
Feature Engineering: Adds TotalSF (total square footage), HouseAge (years since built), and RemodAge (years since remodel).
Evaluation: Uses RMSE and R² metrics to assess model performance.
Visualization: Scatter plot of actual vs. predicted prices.
Deployment: Streamlit app for interactive predictions.

Repository Structure
house_price_project/
├── train_model.py        # Script to train and save the model
├── app.py               # Streamlit app for predictions
├── train.csv            # Kaggle dataset (not tracked in Git)
├── house_price_model.pkl # Trained model (not tracked in Git)
├── actual_vs_predicted.png # Saved visualization
├── requirements.txt      # Python dependencies
├── README.md            # This file

Prerequisites

Python 3.7+
VS Code with Python extension
Git for version control
Kaggle Account to download train.csv

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Install Dependencies:
pip install -r requirements.txt

Dependencies: pandas, numpy, scikit-learn, matplotlib, streamlit.

Download the Dataset:

Download train.csv from Kaggle.
Place it in the house_price_project folder.



Usage

Train the Model:

Run the training script to train the Random Forest model with hyperparameter tuning:python train_model.py


Outputs:
Best hyperparameters
RMSE and R² scores
Sample prediction
Scatter plot (actual_vs_predicted.png)
Saved model (house_price_model.pkl)




Run the Streamlit App Locally:

Start the Streamlit app:streamlit run app.py


Open http://localhost:8501 in your browser.
Enter house features (e.g., TotalSF, LotArea, Neighborhood) and click "Predict Price" to get a prediction.



Deployment on Streamlit Cloud

Push to GitHub:

Initialize Git, add files, and push:git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/house-price-prediction.git
git push -u origin main


Ensure train.csv and house_price_model.pkl are uploaded (or regenerate house_price_model.pkl on Streamlit Cloud).


Deploy:

Go to Streamlit Cloud and sign in with GitHub.
Click "New app" > "From existing repo".
Select your repository, branch (main), and main file (app.py).
Deploy the app. Access it via the provided URL (e.g., https://your-app-name.streamlit.app).



Example Output
Running train_model.py:
Best Parameters: {'regressor__n_estimators': 200, 'regressor__max_depth': 20, 'regressor__min_samples_split': 2}
Root Mean Squared Error: 27000.00
R^2 Score: 0.90
Predicted house price for sample: $175,000.00
Model saved as house_price_model.pkl

Streamlit app: Enter features to get a prediction like $175,000.00.
Future Improvements

Feature Selection: Use feature importance to reduce input complexity.
Advanced Tuning: Implement Bayesian optimization or Random Search.
Cross-Validation: Add k-fold cross-validation for robust evaluation.
Enhanced UI: Add more input fields or visualizations to the Streamlit app.

Troubleshooting

FileNotFoundError: Ensure train.csv and house_price_model.pkl are in the project folder.
ModuleNotFoundError: Run pip install -r requirements.txt.
Streamlit Issues: Verify app.py matches training feature columns.
Deployment Errors: Ensure GitHub repo is public and files are uploaded.

License
MIT License. See LICENSE for details.
Contact
For questions, open an issue or contact [your-email@example.com].