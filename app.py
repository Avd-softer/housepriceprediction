# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.express as px
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore")

# ----------------------------
# NLTK Setup
# ----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    try:
        with open("xgb_multimodal.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        with open("preprocessor.pkl", "rb") as f:
            prep = pickle.load(f)
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        cnn_model = load_model("cnn_text_model.h5", compile=False)
        return xgb_model, prep, tokenizer, cnn_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


xgb_model, prep, tokenizer, cnn_model = load_models()

# ----------------------------
# Initialize stemmer & stopwords
# ----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


# ----------------------------
# Text preprocessing
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SmartPrice", layout="wide")
st.markdown("""
    <style>
    /* Background animation */
    body {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c1ee);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar full height with glass effect */
    [data-testid="stSidebar"] {
        height: 100vh !important;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(255,255,255,0.3);
        padding: 20px;
    }

    /* Section fade-in */
    .section {
        animation: fadeIn 1.5s ease;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(30px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Badge styling */
    .badge {
        display:inline-block;
        background: linear-gradient(45deg,#81d4fa,#0288d1);
        color:white;
        padding:6px 12px;
        margin:4px;
        border-radius:20px;
        font-size:14px;
        transition:all 0.3s ease;
    }
    .badge:hover {
        transform:scale(1.2) rotate(-2deg);
        background:linear-gradient(45deg,#ff7043,#d32f2f);
    }

    /* Cards for outputs */
    .result-card {
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        animation: fadeIn 1s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# App Title
# ----------------------------
st.markdown("<h1 class='section'>🏠 SmartPrice: Ames Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='section'>Predict house prices using <b>structured property data</b> and <b>property descriptions</b> with feature contribution, text insights, and animations.</p>",
    unsafe_allow_html=True)

# ----------------------------
# Check models
# ----------------------------
if xgb_model is None or prep is None or cnn_model is None or tokenizer is None:
    st.error("❌ Failed to load models. Check all model files.")
    st.stop()

# ----------------------------
# User Input
# ----------------------------
st.sidebar.header("📋 Property Details")
st.sidebar.markdown("""
    **Instructions:**
    - Bedrooms: 1–8  
    - Bathrooms: 0–3  
    - Living Area: 334–5642 sq.ft  
    - Basement: 0–6110 sq.ft  
    - Garage Cars: 0–4  
    """)


def get_user_input():
    data = {}
    data['Gr Liv Area'] = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=334, max_value=5642,
                                                  value=1500)
    data['Total Bsmt SF'] = st.sidebar.number_input("Total Basement SF", min_value=0, max_value=6110, value=800)
    data['Overall Qual'] = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
    data['Garage Cars'] = st.sidebar.slider("Garage Cars", 0, 4, 1)
    data['Bedroom AbvGr'] = st.sidebar.slider("Bedrooms Above Ground", 1, 8, 3)
    data['Full Bath'] = st.sidebar.slider("Full Bathrooms", 0, 3, 2)
    house_style_options = ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "2.5Unf", "2.5Fin", "1.5Unf"]
    data['House Style'] = st.sidebar.selectbox("House Style", house_style_options)
    default_desc = "Beautiful home with modern kitchen, hardwood floors, updated bathrooms, central air, finished basement, attached garage."
    data['PropertyDescription'] = st.sidebar.text_area("Property Description", default_desc, height=120)
    return data


user_data = get_user_input()

# ----------------------------
# Build input dataframe
# ----------------------------
structured_cols = prep.feature_names_in_
input_df = pd.DataFrame(columns=structured_cols)

for col in structured_cols:
    if col in user_data:
        input_df.at[0, col] = user_data[col]
    else:
        if prep._feature_names_in_type_dict[col] == 'numeric' if hasattr(prep, '_feature_names_in_type_dict') else True:
            input_df.at[0, col] = 0
        else:
            input_df.at[0, col] = 'None'

input_df['PropertyDescription'] = user_data['PropertyDescription']

# ----------------------------
# Process text
# ----------------------------
input_df['clean_desc'] = input_df['PropertyDescription'].apply(clean_text)
MAXLEN = 100
seq = tokenizer.texts_to_sequences(input_df['clean_desc'])
padded_seq = pad_sequences(seq, maxlen=MAXLEN, padding='post')
text_embeddings = cnn_model.predict(padded_seq, verbose=0)

# ----------------------------
# Structured features
# ----------------------------
X_structured = prep.transform(input_df[structured_cols])

# ----------------------------
# Combine structured + text
# ----------------------------
X_combined = np.hstack([X_structured, text_embeddings])

# ----------------------------
# Price Prediction
# ----------------------------
st.markdown("<h2 class='section'>💰 Price Prediction</h2>", unsafe_allow_html=True)
try:
    pred_price = xgb_model.predict(X_combined)[0]
    inr_price = pred_price * 83

    st.markdown(
        f"<div class='result-card'><b>Predicted Sale Price (USD):</b> <span style='color:green;'>${pred_price:,.2f}</span></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='result-card'><b>Converted Sale Price (INR):</b> <span style='color:purple;'>₹{inr_price:,.2f}</span></div>",
        unsafe_allow_html=True)
except Exception as e:
    st.error(f"Prediction failed: {e}")

# ----------------------------
# Feature Contribution
# ----------------------------
st.markdown("<h2 class='section'>📊 Feature Contribution</h2>", unsafe_allow_html=True)
impact_scores = np.abs(X_structured[0][:7]) * 1000
impact_df = pd.DataFrame({'Feature': structured_cols[:7], 'Estimated Contribution': impact_scores})

fig_bar = px.bar(impact_df, x='Estimated Contribution', y='Feature', color='Estimated Contribution',
                 color_continuous_scale='Plasma', text='Estimated Contribution', orientation='h')
fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
st.plotly_chart(fig_bar, use_container_width=True)

top_n = 5
top_features = impact_df.sort_values(by='Estimated Contribution', ascending=False).head(top_n)
fig_pie = px.pie(top_features, values='Estimated Contribution', names='Feature',
                 color_discrete_sequence=px.colors.qualitative.Set3,
                 title=f"Top {top_n} Feature Contributions")
fig_pie.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * top_n)
st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------
# Animated Scatter Plot
# ----------------------------
st.markdown("<h2 class='section'>📈 Animated Feature vs Price</h2>", unsafe_allow_html=True)
anim_df = pd.DataFrame({'Feature': structured_cols[:7],
                        'Value': np.abs(X_structured[0][:7]),
                        'Price': [pred_price] * 7})
fig_anim = px.scatter(anim_df, x='Value', y='Price', animation_frame='Feature',
                      size='Value', range_y=[0, pred_price * 2],
                      color='Feature', size_max=60, template="plotly_dark")
st.plotly_chart(fig_anim, use_container_width=True)

# ----------------------------
# Text Analysis
# ----------------------------
st.markdown("<h2 class='section'>📝 Property Description Analysis</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Original Description")
    st.write(input_df['PropertyDescription'].iloc[0])
with col2:
    st.subheader("Processed Text")
    processed_text = input_df['clean_desc'].iloc[0]
    st.write(processed_text)
    words = processed_text.split()
    unique_words = set(words)
    st.markdown("**Unique Words:**")
    badges_html = "<div style='display:flex; flex-wrap: wrap;'>"
    for w in unique_words:
        badges_html += f"<span class='badge'>{w}</span>"
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)

    top_words = Counter(words).most_common(10)
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    fig_words = px.bar(top_words_df, x='Word', y='Frequency', color='Frequency',
                       color_continuous_scale='Viridis', text='Frequency', title="Top Words in Description")
    fig_words.update_traces(textposition='outside')
    st.plotly_chart(fig_words, use_container_width=True)

# ----------------------------
# Pricing Guidance
# ----------------------------
st.markdown("<h2 class='section'>💡 Pricing Guidance</h2>", unsafe_allow_html=True)
st.info(
    "- Compare predicted price with neighborhood average  \n- Improve property description to increase perceived value  \n- Adjust property features according to feature contribution charts")

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    "<hr><p style='text-align:center;color:gray;font-size:12px;'>&copy; 2025 SmartPrice by Avd_upadhye. All Rights Reserved.</p>",
    unsafe_allow_html=True)
