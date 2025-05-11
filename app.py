# app.py - Futuristic ML Platform with 4 Themes
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, silhouette_score, 
                           mean_squared_error, r2_score)
import base64

# Configure page
st.set_page_config(
    page_title="NEXUS AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Theme Configurations
THEMES = {
    "Cyber Nexus": {
        "primary": "#00f3ff",
        "secondary": "#7b00ff",
        "bg_color": "#0a0e29",
        "text": "#ffffff",
        "font": "Courier New",
        "model": "Neural Matrix",
        "particles": "✨🌌💫"
    },
    "Quantum Void": {
        "primary": "#ff00ff",
        "secondary": "#00ffff",
        "bg_color": "#000000",
        "text": "#ffffff",
        "font": "Arial Black",
        "model": "Quantum Classifier",
        "particles": "⚛️🌀🌠"
    },
    "Neon Horizon": {
        "primary": "#39ff14",
        "secondary": "#ff073a",
        "bg_color": "#011627",
        "text": "#ffffff",
        "font": "Impact",
        "model": "Plasma Network",
        "particles": "🔆🌃🌉"
    },
    "Galactic Core": {
        "primary": "#ff6b6b",
        "secondary": "#4ecdc4",
        "bg_color": "#0d1b2a",
        "text": "#ffffff",
        "font": "Orbitron",
        "model": "Stellar Engine",
        "particles": "🌌🛸💥"
    }
}

def apply_theme(theme):
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family={theme['font'].replace(' ', '+')}&display=swap');
        
        body, .stApp {{
            background-color: {theme['bg_color']};
            color: {theme['text']};
            font-family: '{theme['font']}', sans-serif;
        }}
        
        .main-container {{
            background: linear-gradient(45deg, {theme['primary']}20, {theme['secondary']}20);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid {theme['primary']}80;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 30px {theme['primary']}40;
        }}
        
        h1, h2, h3 {{
            color: {theme['primary']};
            text-shadow: 0 0 10px {theme['primary']}80;
        }}
        
        .stButton>button {{
            background: {theme['primary']};
            color: {theme['bg_color']};
            border: none;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            transition: all 0.3s;
        }}
        
        .stButton>button:hover {{
            background: {theme['secondary']};
            box-shadow: 0 0 15px {theme['secondary']}80;
        }}
        
        .dataframe {{
            background-color: {theme['bg_color']} !important;
            color: {theme['text']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

def init_session():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'features' not in st.session_state:
        st.session_state.features = []
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'step' not in st.session_state:
        st.session_state.step = 1

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Dataset needs at least 2 numeric columns")
            return None
            
        st.session_state.data = df
        st.session_state.step = 2
        st.success(f"🌀 Data Matrix Initialized ({len(df)} records)")
        return df
    except Exception as e:
        st.error(f"Data Decryption Failed: {str(e)}")
        return None

def analyze_data(theme):
    st.header(f"🔍 {theme['model']} Analysis")
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Data Signature")
        st.dataframe(df.describe().style.format("{:.2f}"), height=300)
        
    with col2:
        st.write("### Quantum Correlation")
        corr = df.corr()
        fig = px.imshow(corr, text_auto=".2f", 
                       color_continuous_scale=[theme['primary'], theme['secondary']])
        fig.update_layout(
            plot_bgcolor=theme['bg_color'],
            paper_bgcolor=theme['bg_color'],
            font_color=theme['text']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button(f"⚡ Activate {theme['model']}", key="train_btn"):
        st.session_state.step = 3

def train_model(theme):
    st.header(f"🚀 {theme['model']} Training")
    df = st.session_state.data
    X = df[st.session_state.features]
    y = df[st.session_state.target]
    
    if y.nunique() < 5:  # Classification
        model = LogisticRegression(max_iter=1000)
        model_type = "classifier"
    else:  # Regression
        model = LinearRegression()
        model_type = "regressor"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    with st.spinner(f"Initializing {theme['model']}..."):
        model.fit(X_train, y_train)
        st.session_state.model = model
        y_pred = model.predict(X_test)
        
        if model_type == "classifier":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"**Quantum Lock Achieved** | Accuracy: {acc:.2f}")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.success(f"**Gravitational Sync Complete** | RMSE: {rmse:.2f}")
        
        # Jet animation effect
        st.markdown(f"""
        <div style="text-align: center; font-size: 40px; margin: 20px 0;">
            🛩️🛩️🛩️
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.step = 4

def show_results(theme):
    st.header("📊 Holographic Results")
    df = st.session_state.data
    model = st.session_state.model
    
    # 3D Visualization
    fig = px.scatter_3d(df, 
                        x=st.session_state.features[0],
                        y=st.session_state.features[1],
                        z=st.session_state.target,
                        color=st.session_state.target,
                        color_continuous_scale=[theme['primary'], theme['secondary']])
    
    fig.update_layout(
        scene=dict(
            xaxis_title=st.session_state.features[0],
            yaxis_title=st.session_state.features[1],
            zaxis_title=st.session_state.target,
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font_color=theme['text']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    if hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'Feature': st.session_state.features,
            'Impact': np.abs(model.coef_[0])
        }).sort_values('Impact', ascending=False)
        
        fig = px.bar(importance, 
                     x='Impact', 
                     y='Feature', 
                     orientation='h',
                     color='Impact',
                     color_continuous_scale=[theme['primary'], theme['secondary']])
        
        fig.update_layout(
            title='Quantum Feature Resonance',
            plot_bgcolor=theme['bg_color'],
            paper_bgcolor=theme['bg_color'],
            font_color=theme['text']
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    init_session()
    
    # Theme Selection
    with st.sidebar:
        st.title("🌌 Theme Matrix")
        theme_name = st.selectbox("", list(THEMES.keys()))
        theme = THEMES[theme_name]
        apply_theme(theme)
        
        st.markdown("---")
        st.header("Data Port")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
        
        if uploaded_file and st.session_state.step == 1:
            load_data(uploaded_file)
    
    # Main Interface
    st.markdown(f"<div class='main-container'>", unsafe_allow_html=True)
    
    if st.session_state.step == 1:
        st.title("NEXUS AI Platform")
        st.markdown(f"""
        <div style="text-align: center; margin: 5rem 0;">
            <h1 style="font-size: 4em;">{theme['particles']}</h1>
            <h3>Upload Dataset to Initiate Quantum Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
    elif st.session_state.step == 2:
        df = st.session_state.data
        st.title("Quantum Data Matrix")
        
        with st.expander("Feature Selection", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.target = st.selectbox("Target Variable", df.columns)
            with cols[1]:
                st.session_state.features = st.multiselect("Features", 
                                                          [c for c in df.columns if c != st.session_state.target],
                                                          default=df.columns[0])
                
        analyze_data(theme)
        
    elif st.session_state.step == 3:
        train_model(theme)
        
    elif st.session_state.step == 4:
        show_results(theme)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
