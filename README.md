# 🌸 Iris Flower Classification

This project is a web application built using **Streamlit** and **Scikit-learn** that predicts the species of an Iris flower based on four flower measurements.

I created this project to learn how to deploy a machine learning model as an interactive web application. Along the way, I learned how to serialize models using Joblib and Pickle, build a user-friendly interface with Streamlit, and display prediction confidence scores.

## Features

- Predicts Iris flower species
- Interactive sliders for feature input
- Confidence score for each prediction
- Supports both Joblib and Pickle model formats
- Clean and responsive Streamlit interface

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
app.py
models/
README.md
requirements.txt
```

## What I Learned

- Building machine learning web applications with Streamlit
- Saving and loading models using Joblib and Pickle
- Working with Streamlit widgets and session state
- Handling errors and organizing project files

## Future Improvements

- Deploy the application online
- Add visualizations of the Iris dataset
- Display feature importance
- Improve the UI with additional styling
