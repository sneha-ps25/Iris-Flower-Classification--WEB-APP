# Iris Flower Classifier

A Streamlit web application that predicts the species of an Iris flower from user-provided sepal and petal measurements, using a pre-trained Random Forest classifier.

## Features

- Interactive sliders for sepal length, sepal width, petal length, and petal width
- Selectable model format at runtime: joblib or pickle
- On-demand model reload without restarting the app
- Sidebar display of model metadata (type, accuracy, feature count, class count)
- Confidence score visualization for each predicted class
- Fallback to default values if model or configuration files are missing

## Project Structure

```

├── app.py                   
├── models/
│   ├── iris_model.joblib    
│   ├── iris_model.pickle   
│   ├── model_info.json
│   └── feature_ranges.json  
├── requirements.txt
└── README.md
```

## Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
```

## Installation

```bash
python -m venv venv
Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Ensure the `models/` directory contains `iris_model.joblib`, `iris_model.pickle`, `model_info.json`, and `feature_ranges.json`, then run:

```bash
streamlit run app.py
```
Dataset : 
      The app uses the Iris Dataset, it is one of the most well-known datasets in machine learning.
----------------------------------------------------      
| Property       | Details                         |
|----------------|---------------------------------|
| Total Samples  | 150 (50 per class)              |
| Features       | 4                               |
| Classes        | 3                               |
| Missing Values | None                            |
----------------------------------------------------
The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (Fisher, 1936) contains 150 samples across 3 species — *Setosa*, *Versicolor*, *Virginica* — with 4 measured features per sample.

Input Features :
---------------------------------------------------
| Feature      | Description               | Unit |
|--------------|---------------------------|------|
| Sepal Length | Length of the sepal       | cm   |
| Sepal Width  | Width of the sepal        | cm   |
| Petal Length | Length of the petal       | cm   |
| Petal Width  | Width of the petal        | cm   |
---------------------------------------------------

Target Classes :
----------------------------
| Label | Species          |
|-------|------------------|
| 0     | Iris Setosa      |
| 1     | Iris Versicolor  |
| 2     | Iris Virginica   |
----------------------------

Machine Learning Model :
-------------------------------------------------
| Property         | Details                    |
|------------------|----------------------------|
| Algorithm        | Random Forest Classifier   |
| Number of Trees  | 100                        |
| Train/Test Split | 80% / 20%                  |
| Random State     | 42                         |
| Accuracy         | ~96% to 100%               |
| Saved Formats    | joblib and pickle          |
-------------------------------------------------

## Built With

Streamlit · scikit-learn · pandas · NumPy

