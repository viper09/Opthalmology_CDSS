# app.py - Enhanced UI Version

import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image # For robust image handling
import plotly.express as px # For interactive plots
import plotly.graph_objects as go # For advanced plot customization
import time # For simulating processing time with sleep

# --- Custom CSS for Enhanced UI ---
def load_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #d1f3d8 100%); /* Changed to a subtle light green */
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
        color: #2c3e50; /* Set text color for readability */
    }
    
    .info-card h3, .info-card h4, .info-card p, .info-card small,
    .info-card .stSlider, .info-card .stSelectbox, .info-card .stRadio {
        color: #2c3e50; /* Ensure all text within info-card is dark for contrast */
    }
    /* Specific Streamlit elements within info-card that might need color override */
    /* Original dark text, ensure it remains dark or adjust as needed for info cards */
    .stSlider > div > div > div > div { color: #2c3e50 !important; } /* Slider value */
    .stSelectbox > div > div > div > div { color: #2c3e50 !important; } /* Selectbox selected value */
    .stRadio > div > label > div { color: #2c3e50 !important; } /* Radio button labels */
    
    /* NEW CSS FOR WHITE TEXT ON DARK BACKGROUND */
    /* General text color for input labels and selectbox values globally */
    .stTextInput label, .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label,
    .st-b3, /* Text in selectbox when an option is chosen */
    .st-b1, /* More general text like default labels */
    .st-be, /* Placeholder/selected value text in inputs */
    .st-bd, /* Options in dropdown */
    .st-cg, /* Options in dropdown */
    .st-br /* Selected value in selectbox */
    {
        color: white !important;
    }

    /* To make option text white in the dropdown list itself (when opened) */
    div[role="listbox"] div span {
        color: white !important;
    }

    /* Ensure the selected option in the selectbox input field is white */
    .st-ck { /* This targets the actual display area of the selected item in selectbox */
        color: white !important;
    }
    .st-ch { /* This also affects the selected value in the display */
        color: white !important;
    }

    /* Change color of numbers in slider (e.g., '50' for age) when not inside .info-card */
    .stSlider div > div > div > div > div[data-testid="stTickValue"] {
        color: white !important;
    }
    /* --- END NEW CSS FOR WHITE TEXT ON DARK BACKGROUND --- */


    .result-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .prediction-result {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    
    /* Image upload area */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #e0f2f7 0%, #b2ebf2 100%); /* Changed to a light blue/teal */
        margin: 1rem 0;
        color: #2c3e50; /* Ensure text is dark for readability */
    }

    .upload-area h4, .upload-area p, .upload-area small {
        color: #2c3e50; /* Ensure all text within upload-area is dark for contrast */
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #e8f5e9 0%, #d1f3d8 100%); /* Changed to a subtle light green */
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem;
        color: #2c3e50; /* Set text color for readability */
    }
    
    /* Warning and info boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* General app text and header color to white */
    body, .stApp {
        color: white; 
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced Header Component ---
def display_header():
    st.markdown("""
    <div class="main-header">
        <h1>👁️ EyeAI Clinical Assistant</h1>
        <p>Advanced AI-Powered Ophthalmology Diagnosis Support</p>
    </div>
    """, unsafe_allow_html=True)

# --- Patient Info Card Component ---
def patient_info_card():
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=1, max_value=100, value=50, help="Patient's age in years")
    
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"], help="Patient's biological sex")
    
    with col3:
        eye_side = st.selectbox("Eye Side", ["Left", "Right"], help="Which eye is being examined")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return age, sex, eye_side

# --- Image Upload Component ---
def image_upload_component():
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Fundus Image Upload")
    
    st.markdown("""
    <div class="upload-area">
        <h4>📁 Upload Eye Fundus Image</h4>
        <p>Supported formats: JPG, JPEG, PNG</p>
        <p><small>For best results, use high-quality fundus photographs</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an eye fundus image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear fundus photograph for analysis"
    )
    
    if uploaded_file is not None:
        # Display image with enhanced styling
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fundus Image", use_container_width=True) # Fixed parameter
            
            # Image info
            st.info(f"📋 Image Details: {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return uploaded_file

# --- Prediction Results Component ---
def display_prediction_results(predicted_diagnosis_full, prediction_proba, label_encoder):
    st.markdown("""
    <div class="result-card">
        <div class="prediction-result">🎯 Diagnosis: {}</div>
        <div class="confidence-text">Confidence: {:.1f}%</div>
    </div>
    """.format(predicted_diagnosis_full, max(prediction_proba) * 100), unsafe_allow_html=True)
    
    # Create probability chart
    proba_df = pd.DataFrame({
        'Diagnosis': label_encoder.classes_, # Directly use label_encoder.classes_
        'Probability': prediction_proba
    }).sort_values(by='Probability', ascending=False)
    
    # Interactive bar chart
    fig = px.bar(
        proba_df, 
        x='Probability', 
        y='Diagnosis',
        orientation='h',
        title="📊 Prediction Confidence by Diagnosis",
        color='Probability',
        color_continuous_scale='Viridis',
        text='Probability'
    )
    
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed probability table
    st.markdown("### 📋 Detailed Results")
    proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{x:.3f}")
    proba_df['Confidence %'] = proba_df['Probability'].apply(lambda x: f"{float(x)*100:.1f}%")
    
    st.dataframe(
        proba_df[['Diagnosis', 'Confidence %']], 
        use_container_width=True,
        hide_index=True
    )

# --- Sidebar Information ---
def display_sidebar():
    with st.sidebar:
        st.markdown("## ℹ️ About EyeAI")
        st.markdown("""
        This AI system analyzes fundus images to assist in diagnosing:
        
        🔹 **Normal** - Healthy eye condition  
        🔹 **Diabetic Retinopathy** - Diabetes-related eye damage  
        🔹 🔹 **Glaucoma** - Optic nerve damage  
        🔹 **Cataract** - Lens clouding  
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        
        # Actual performance metrics from your training
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "0.5666") # Updated with actual value
        with col2:
            st.metric("Precision", "0.7205") # Updated with actual value
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Notice")
        st.warning("""
        This tool is for **clinical decision support only**. 
        Always consult with qualified ophthalmologists for final diagnosis and treatment decisions.
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.info("For technical support or questions, contact the development team.")

# --- Helper Function for Feature Extraction (Moved to global scope) ---
# This function must be defined BEFORE main() as it's called within main().
def extract_image_features_for_prediction(uploaded_file, feature_extractor_model):
    """Processes an uploaded image file and extracts features using the pre-trained CNN."""
    try:
        # Use PIL for robust image opening directly from Streamlit's UploadedFile object
        img_pil = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img_pil) # PIL to numpy array (RGB)

        img_resized = cv2.resize(img_np, TARGET_SIZE) # Resize (img_np is RGB already)
        img_array = np.expand_dims(img_resized, axis=0) # Add batch dimension
        img_array = preprocess_input(img_array) # Preprocess for ResNet (handles normalization)

        features = feature_extractor_model.predict(img_array, verbose=0)
        return features.flatten() # Return 1D feature vector
    except Exception as e:
        st.error(f"❌ Error processing image or extracting features: {e}")
        return None

# --- Streamlit App Configuration (Single, Correct Place) ---
st.set_page_config(
    page_title="EyeAI Clinical Assistant", 
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# --- Configuration (Moved here for clarity) ---
MODEL_PATH = 'catboost_multi_class_odir_model.cbm'

IMG_HEIGHT = 224
IMG_WIDTH = 224
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# LabelEncoder and Diagnosis Mapping are now created directly in app.py as per your prepare_odir_data.py
DIAGNOSIS_CLASSES = ['Cataract', 'Diabetes', 'Glaucoma', 'Normal']
label_encoder = LabelEncoder()
label_encoder.fit(DIAGNOSIS_CLASSES)

# Since label_encoder.classes_ directly provides the full names, we don't need a separate joblib-loaded mapping.
# We'll create a simple mapping dictionary based on the fixed order for clarity in display.
diagnosis_mapping = {
    label_encoder.classes_[i]: label_encoder.classes_[i] for i in range(len(label_encoder.classes_))
}
# If you had distinct short codes (e.g., 'C') that mapped to full names (e.g., 'Cataract'),
# this dictionary would be manually defined here:
# diagnosis_mapping = {
#    'C': 'Cataract',
#    'D': 'Diabetes',
#    'G': 'Glaucoma',
#    'N': 'Normal'
# }
# But given label_encoder.classes_ are the full names, this is simpler.


# --- Load Models (Cached) ---
@st.cache_resource
def load_catboost_model_cached(model_path):
    """Loads the pre-trained CatBoost model."""
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading CatBoost model: {e}")
        return None

@st.cache_resource
def load_feature_extractor_cached(img_height, img_width):
    """Loads the pre-trained ResNet50 model for feature extraction."""
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        feature_extractor = Model(inputs=base_model.input,
                                   outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
        return feature_extractor
    except Exception as e:
        st.error(f"❌ Error loading ResNet50 feature extractor: {e}")
        return None

# --- Main App Interface ---
def main():
    # Load models globally once
    with st.spinner("🔄 Loading AI models..."):
        catboost_model = load_catboost_model_cached(MODEL_PATH)
        resnet_feature_extractor = load_feature_extractor_cached(IMG_HEIGHT, IMG_WIDTH)

    if catboost_model is None or resnet_feature_extractor is None:
        st.error("❌ Failed to load required AI assets. Please check your model files.")
        st.stop() # Stop the app if essential assets are missing

    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Patient information
        age, sex, eye_side = patient_info_card()
        
        # Image upload
        uploaded_file = image_upload_component()
        
        # Prediction button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔍 Analyze Image & Get Diagnosis", use_container_width=True)
        
        # Prediction logic
        if predict_button:
            if uploaded_file is None:
                st.warning("⚠️ Please upload an eye fundus image to get a prediction. �")
            else:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Image processing
                status_text.text("🔄 Processing image...")
                progress_bar.progress(25)
                time.sleep(0.5) # Simulate processing time
                
                image_features = extract_image_features_for_prediction(uploaded_file, resnet_feature_extractor)
                
                if image_features is None:
                    st.error("❌ Could not extract features from the image. Prediction aborted.")
                    status_text.empty() # Clear status
                    progress_bar.empty() # Clear progress
                else:
                    # Step 2: Preparing data
                    status_text.text("🔄 Preparing data for analysis...")
                    progress_bar.progress(50)
                    time.sleep(0.5) # Simulate processing time
                    
                    # Create dictionary to hold the input data for the single prediction
                    input_data_dict = {}
                    input_data_dict['age'] = age
                    # Map sex and eye_side to match training data format (0/1 for sex, 'left'/'right' for eye_side)
                    input_data_dict['sex'] = 0 if sex == 'Male' else 1
                    input_data_dict['eye_side'] = 'left' if eye_side == 'Left' else 'right'

                    # Add image features
                    for i, feature_val in enumerate(image_features):
                        input_data_dict[f'img_feat_{i}'] = feature_val

                    # Convert to DataFrame
                    input_df = pd.DataFrame([input_data_dict]) # Pass as list of dicts to create single row

                    # Ensure column order matches training, especially for categorical features
                    if hasattr(catboost_model, 'feature_names_') and catboost_model.feature_names_:
                        expected_columns = catboost_model.feature_names_
                    else:
                        # Fallback: manually construct the order based on your train_multi_class_model.py's X_columns
                        expected_columns = ['age', 'sex', 'eye_side'] + [f'img_feat_{i}' for i in range(image_features.shape[0])]
                    
                    input_df_reindexed = input_df.reindex(columns=expected_columns, fill_value=0) # Use fill_value=0 for safety

                    # Step 3: Making prediction
                    status_text.text("🤖 AI is analyzing the image...")
                    progress_bar.progress(75)
                    time.sleep(1) # Simulate processing time
                    
                    prediction_encoded = catboost_model.predict(input_df_reindexed).flatten()[0]
                    prediction_proba = catboost_model.predict_proba(input_df_reindexed).flatten()
                    
                    # Decode prediction directly using label_encoder
                    predicted_diagnosis_full = label_encoder.inverse_transform([prediction_encoded])[0]

                    # Step 4: Complete
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display results
                    display_prediction_results(predicted_diagnosis_full, prediction_proba, label_encoder) # Removed diagnosis_mapping as argument
    
    with col2:
        # Quick tips and information
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        **For Best Results:**
        
        ✅ Use high-resolution images  
        ✅ Ensure good lighting  
        ✅ Center the optic disc  
        ✅ Minimize blur/artifacts  
        
        **Image Quality Checklist:**
        
        📋 Clear optic disc visible  
        📋 Macula clearly defined  
        📋 Blood vessels distinct  
        📋 No significant glare  
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # The "Today's Statistics" section has been removed as per your request.

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🏥 Built with ❤️ for Ophthalmology Clinical Decision Support</p>
    <p><small>Version 2.0 | Enhanced UI | AI-Powered Diagnosis</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

�
