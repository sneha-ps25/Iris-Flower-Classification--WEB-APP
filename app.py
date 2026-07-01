import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
   .block-container{
            padding-top: 1rem !important;
            }
    .main-header {
        font-size: 3rem;
        color: #6a0dad;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0rem;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #6a0dad;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        text-align: center;
        color: white;
        font-weight: bold;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Loaders --------------------
DEFAULT_FEATURE_RANGES = {
    'sepal_length': {'min': 4.0, 'max': 8.0, 'default': 5.8},
    'sepal_width': {'min': 2.0, 'max': 4.5, 'default': 3.0},
    'petal_length': {'min': 1.0, 'max': 7.0, 'default': 4.0},
    'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2},
}

DEFAULT_MODEL_INFO = {
    'model_type': 'RandomForest',
    'accuracy': 0.96,
    'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target_names': ['setosa', 'versicolor', 'virginica'],
}


@st.cache_resource(show_spinner=False)
def load_model(format_type: str):
    """Load the model from the specified format."""
    try:
        if format_type == 'joblib':
            path = 'models/iris_model.joblib'
            if not os.path.exists(path):
                return None
            return joblib.load(path)
        elif format_type == 'pickle':
            path = 'models/iris_model.pickle'
            if not os.path.exists(path):
                return None
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_model_info():
    """Load model metadata, falling back to defaults if unavailable."""
    try:
        with open('models/model_info.json', 'r') as f:
            return json.load(f)
    except Exception:
        return DEFAULT_MODEL_INFO


@st.cache_resource(show_spinner=False)
def load_feature_ranges():
    """Load feature ranges for sliders, filling in any missing keys with defaults."""
    try:
        with open('models/feature_ranges.json', 'r') as f:
            loaded = json.load(f)
    except Exception:
        return DEFAULT_FEATURE_RANGES

    # Merge with defaults so missing 'min'/'max'/'default' keys don't crash the app
    merged = {}
    for feature, default_range in DEFAULT_FEATURE_RANGES.items():
        feature_range = loaded.get(feature, {})
        merged[feature] = {
            'min': feature_range.get('min', default_range['min']),
            'max': feature_range.get('max', default_range['max']),
            'default': feature_range.get('default', default_range['default']),
        }
    return merged


# -------------------- Session State --------------------
if 'model_format' not in st.session_state:
    st.session_state.model_format = 'joblib'

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"],
        help="Choose which serialized model format to use for predictions"
    )
    st.session_state.model_format = model_format

    if st.button("🔄 Reload Model"):
        load_model.clear()  # clear cache so the file is actually re-read
        reloaded_model = load_model(model_format)
        if reloaded_model is not None:
            st.success(f"Model loaded from {model_format} format!")
        else:
            st.error(f"Could not load model from {model_format} format. "
                      f"Check that the file exists in the 'models/' folder.")

    st.divider()

    st.subheader("📊 Model Information")
    model_info = load_model_info()
    if model_info:
        st.write(f"**Type:** {model_info.get('model_type', 'RandomForest')}")
        st.write(f"**Accuracy:** {model_info.get('accuracy', 0.96):.1%}")
        st.write(f"**Features:** {len(model_info.get('feature_names', []))}")
        st.write(f"**Classes:** {len(model_info.get('target_names', []))}")

    st.divider()
    st.subheader("🚀 Quick Actions")
    st.session_state.show_dataset_info = st.button("📊 Show Dataset Info")

# -------------------- Load Data --------------------
feature_ranges = load_feature_ranges()
model = load_model(st.session_state.model_format)

# -------------------- Header --------------------
st.markdown('<h1 class="main-header">🌸 Iris Flower Classification</h1>', unsafe_allow_html=True)

st.markdown("""
This app predicts the species of an Iris flower based on its measurements using a machine
learning model. Adjust the sliders below to input the flower's characteristics and see the
prediction!
""")



# -------------------- Input Form --------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Features")

    sepal_length = st.slider(
        "**Sepal Length (cm)**",
        min_value=float(feature_ranges['sepal_length']['min']),
        max_value=float(feature_ranges['sepal_length']['max']),
        value=float(feature_ranges['sepal_length']['default']),
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    sepal_width = st.slider(
        "**Sepal Width (cm)**",
        min_value=float(feature_ranges['sepal_width']['min']),
        max_value=float(feature_ranges['sepal_width']['max']),
        value=float(feature_ranges['sepal_width']['default']),
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    petal_length = st.slider(
        "**Petal Length (cm)**",
        min_value=float(feature_ranges['petal_length']['min']),
        max_value=float(feature_ranges['petal_length']['max']),
        value=float(feature_ranges['petal_length']['default']),
        step=0.1,
        help="Length of the petal in centimeters"
    )
    petal_width = st.slider(
        "**Petal Width (cm)**",
        min_value=float(feature_ranges['petal_width']['min']),
        max_value=float(feature_ranges['petal_width']['max']),
        value=float(feature_ranges['petal_width']['default']),
        step=0.1,
        help="Width of the petal in centimeters"
    )

with col2:
    st.header("📊 Current Values")
    features_df = pd.DataFrame({
        'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(features_df, hide_index=True, use_container_width=True)

input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# -------------------- Prediction --------------------
if st.button("🎯 Predict Species", type="primary", use_container_width=True):
    if model is not None and model_info is not None:
        try:
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)[0]
            predicted_class = model_info['target_names'][prediction[0]]

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Prediction Result")
            st.markdown(f"**Predicted Species:** **{predicted_class}**")

            st.markdown("### 📈 Confidence Scores")
            for i, prob in enumerate(prediction_proba):
                species = model_info['target_names'][i]
                percentage = prob * 100
                col_prog, col_text = st.columns([3, 1])
                with col_prog:
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {percentage}%;">
                            {percentage:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_text:
                    st.write(f"**{species}**")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
    else:
        st.error("❌ Model could not be loaded. Please check if the model files exist.")

# -------------------- About Section --------------------
with st.expander("📚 About the Iris Dataset"):
    st.markdown("""
    The Iris flower dataset is a classic dataset in machine learning and statistics,
    introduced by Ronald Fisher in 1936.

    **Dataset Characteristics:**
    - 150 samples (50 per class)
    - 4 features per sample
    - 3 classes (species)

    **Species:**
    - **Iris Setosa**
    - **Iris Versicolor**
    - **Iris Virginica**

    **Features:**
    1. Sepal length (cm)
    2. Sepal width (cm)
    3. Petal length (cm)
    4. Petal width (cm)

    This model uses a **Random Forest classifier** with an accuracy of approximately 96%.
    """)


st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
