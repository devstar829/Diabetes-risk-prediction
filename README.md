 Diabetes Risk Prediction

The Diabetes Risk Prediction system is a tool designed to assess an individual's likelihood of developing diabetes based on various health metrics and lifestyle factors. By analyzing data such as age, blood glucose levels, BMI, and other factors, the system provides personalized risk assessments and recommendations for prevention.

## Overview

This project is organized into several parts:

- **Data Processing & Visualization**  
  The data is sourced from [data/diabetes.csv](data/diabetes.csv) and preprocessed in [`src/train_model.py`](src/train_model.py). A bar chart visualizing the diabetes outcome distribution is generated and saved to [static/chart.png](static/chart.png), then rendered in the [templates/index.html](templates/index.html) file.

- **Machine Learning Model**  
  A **RandomForestClassifier** is trained using scaled features, and both the model and scaler are saved as [src/model/rf_model.pkl](src/model/rf_model.pkl) and [src/model/scaler.pkl](src/model/scaler.pkl) respectively.

- **API Service**  
  The [FastAPI](https://fastapi.tiangolo.com/) based API is implemented in [`api/main.py`](api/main.py). This renders the result visualization and serves static files. To run the API, simply launch the provided FastAPI server.

- **Command-Line Prediction**  
  The [`src/predict.py`](src/predict.py) script allows for console-based predictions where a user inputs health parameters and the system outputs whether the patient is predicted as diabetic or not.

## Installation

1. **Clone the Repository**

    ```sh
    git clone <repository-url>
    cd Diabetes-risk-prediction
    ```

2. **Set Up a Virtual Environment**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    Make sure you have the required packages installed. For example, install dependencies such as `pandas`, `scikit-learn`, `joblib`, `fastapi`, `uvicorn`, and `matplotlib`:

    ```sh
    pip install pandas scikit-learn joblib fastapi uvicorn matplotlib
    ```

## Workflow

1. **Data Preparation & Model Training**

    Run the training script to preprocess data, train the model, and generate the visualization:

    ```sh
    python src/train_model.py
    ```

    This script reads data from [data/diabetes.csv](data/diabetes.csv), scales the features, splits the dataset, trains a Random Forest model, prints the model accuracy, and saves both the model and scaler in the [src/model](src/model/) directory. It also generates and saves a chart in [static/chart.png](static/chart.png).

2. **Running the API**

    Launch the FastAPI application:

    ```sh
    uvicorn api.main:app --reload
    ```

    Then open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the visualization rendered by [templates/index.html](templates/index.html).

3. **Making Predictions from the Command Line**

    Run the prediction script to provide patient data and receive a risk prediction:

    ```sh
    python src/predict.py
    ```

    Follow the prompts to input patient data. The script will output whether the patient is predicted as diabetic or not.

## Additional Notes

- Data file: [data/diabetes.csv](data/diabetes.csv)  
- Visualization: Saved as [static/chart.png](static/chart.png) and rendered in [templates/index.html](templates/index.html)  
- API Entry Point: [`api/main.py`](api/main.py)  
- Model Training and Prediction: [`src/train_model.py`](src/train_model.py) and [`src/predict.py`](src/predict.py)

Happy Coding!