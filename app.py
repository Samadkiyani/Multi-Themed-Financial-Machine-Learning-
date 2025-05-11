import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import yfinance as yf

# Configure page
st.set_page_config(
    page_title="NEXUS AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Configurations
THEMES = {
    "Zombie Theme": {
        "primary": "#ff0000",
        "secondary": "#000000",
        "bg_color": "#2e2e2e",
        "text": "#ffffff",
        "font": "Creepster",
        "model": "Zombie Predictor",
        "particles": "🧟‍♂️🧟‍♀️💀",
        "background": "https://images.unsplash.com/photo-1517841905240-472988babdf9"  # Zombie theme background
    },
    "Futuristic Theme": {
        "primary": "#00f3ff",
        "secondary": "#7b00ff",
        "bg_color": "#0a0e29",
        "text": "#ffffff",
        "font": "Courier New",
        "model": "Neural Matrix",
        "particles": "✨🌌💫",
        "background": "https://images.unsplash.com/photo-1512561861162-0a1c1c3c7d8c"  # Futuristic theme background
    },
    "Game of Thrones Theme": {
        "primary": "#ffcc00",
        "secondary": "#ff3300",
        "bg_color": "#1a1a1a",
        "text": "#ffffff",
        "font": "Cinzel",
        "model": "Westeros Predictor",
        "particles": "🔥❄️⚔️",
        "background": "https://images.unsplash.com/photo-1593642632781-0c3d8b5e6c4f"  # GOT theme background
    },
    "Gaming Theme": {
        "primary": "#39ff14",
        "secondary": "#ff073a",
        "bg_color": "#011627",
        "text": "#ffffff",
        "font": "Press Start 2P",
        "model": "Pixel Predictor",
        "particles": "🎮👾🕹️",
        "background": "https://images.unsplash.com/photo-1511907112600-1b1f5c4c8f1e"  # Gaming theme background
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
            background-image: url('{theme['background']}');
            background-size: cover;
            background-position: center;
        }}
        
        .main-container {{
            background: rgba(255, 255, 255, 0.8);
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
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'steps' not in st.session_state:
        st.session_state.steps = {
            'loaded': False,
            'processed': False,
            'ready_for_model': False,
            'trained': False
        }

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
        st.session_state.steps['loaded'] = True
        st.success(f"🌀 Data Matrix Initialized ({len(df)} records)")
        return df
    except Exception as e:
        st.error(f"Data Loading Failed: {str(e)}")
        return None

def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, period="1y")
        if df.empty:
            st.error("No data found for this ticker.")
            return None
        st.session_state.data = df
        st.session_state.steps['loaded'] = True
        st.success(f"📈 Stock Data for {ticker} Loaded ({len(df)} records)")
        return df
    except Exception as e:
        st.error(f"Failed to fetch stock data: {str(e)}")
        return None

def show_data_preview(df):
    st.write("### Data Preview")
    st.dataframe(df.head())

def feature_selection(df):
    with st.expander("Feature Selection", expanded=True):
        st.session_state.target = st.selectbox("Target Variable", df.columns)
        st.session_state.features = st.multiselect("Features", 
                                                  [c for c in df.columns if c != st.session_state.target],
                                                  default=df.columns[0])

def data_analysis(df, features, target, theme):
    st.header(f"🔍 {theme['model']} Analysis")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.error("Dataset needs at least 2 numeric columns for correlation analysis.")
        return

    if numeric_df.isnull().values.any():
        st.warning("NaN values found in the dataset. Filling NaN values with zeros.")
        numeric_df = numeric_df.fillna(0)

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Data Signature")
        st.dataframe(numeric_df.describe().style.format("{:.2f}"), height=300)
        
    with col2:
        st.write("### Quantum Correlation")
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", 
                       color_continuous_scale=[theme['primary'], theme['secondary']])
        fig.update_layout(
            plot_bgcolor=theme['bg_color'],
            paper_bgcolor=theme['bg_color'],
            font_color=theme['text']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.steps['processed'] = True

def train_model(features, target, model_type, test_size=0.2):
    st.header("🚀 Model Training")
    df = st.session_state.data
    X = df[features]
    y = df[target]
    
    if model_type == "Logistic Regression" and y.nunique() < 5:  # Classification
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Linear Regression":  # Regression
        model = LinearRegression()
    else:  # K-Means Clustering
        model = KMeans(n_clusters=3)
        st.session_state.model = model.fit(X)
        st.session_state.predictions = model.predict(X)
        st.success("K-Means Clustering model trained successfully.")
        st.session_state.steps['trained'] = True
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    with st.spinner(f"Training {model_type}..."):
        model.fit(X_train, y_train)
        st.session_state.model = model
        y_pred = model.predict(X_test)
        
        if model_type == "Logistic Regression":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"**Quantum Lock Achieved** | Accuracy: {acc:.2f}")
        else:  # Linear Regression
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.success(f"**Gravitational Sync Complete** | RMSE: {rmse:.2f}")
        
        st.session_state.predictions = y_pred
        st.session_state.steps['trained'] = True

def evaluate_model(predictions, features, model_type, theme):
    st.header("📊 Evaluation Results")
    if model_type == "K-Means Clustering":
        st.write("### Clustering Results")
        df = st.session_state.data
        df['Cluster'] = predictions
        fig = px.scatter(df, x=features[0], y=features[1], color='Cluster', 
                         color_continuous_scale=[theme['primary'], theme['secondary']])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("### Predictions")
        st.dataframe(pd.DataFrame(predictions, columns=["Predictions"]))
    
    st.session_state.steps['ready_for_model'] = False

def main():
    init_session()
    
    # Theme Selection
    with st.sidebar:
        st.title("🌌 Theme Matrix")
        theme_name = st.selectbox("", list(THEMES.keys()))
        theme = THEMES[theme_name]
        apply_theme(theme)
        
        st.markdown("---")
        st.header("Data Sources")
        data_source = st.radio("Select Data Source", ["Upload Dataset", "Fetch Stock Data"])
        uploaded_file = st.file_uploader("Upload Financial Dataset from Kragle", type=["csv", "xlsx"])
        ticker = st.text_input("Fetch Stock Data using Yahoo Finance (ticker symbol):")
        
        if uploaded_file and not st.session_state.steps['loaded']:
            load_data(uploaded_file)
        elif ticker and not st.session_state.steps['loaded']:
            fetch_stock_data(ticker)

    # Main Interface
    st.markdown(f"<div class='main-container'>", unsafe_allow_html=True)
    
    # Step 1: Data Upload
    st.header("1. Data Upload & Selection")
    if data_source == "Upload Dataset":
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                show_data_preview(df)
                feature_selection(df)
        else:
            st.markdown(f"""
            <div class='feature-selector'>
            📁 **How to Use:**
            1. Upload any CSV or Excel file with numeric data  
            2. Or fetch stock data from Yahoo Finance
            3. Select target variable (what you want to predict)  
            4. Choose features (variables used for prediction)  
            5. The system will automatically handle the rest  
            </div>
            """, unsafe_allow_html=True)
    else:
        if st.session_state.steps['loaded']:
            df = st.session_state.data
            show_data_preview(df)
            feature_selection(df)

    # Step 2: Data Analysis
    if st.session_state.steps['loaded'] and not st.session_state.steps['processed']:
        data_analysis(
            st.session_state.data,
            st.session_state.features,
            st.session_state.target,
            theme
        )

    # Step 3: Model Training
    model_type = st.selectbox("Select Model Type", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])
    if st.session_state.steps['processed'] and not st.session_state.steps['trained']:
        train_model(
            st.session_state.features,
            st.session_state.target,
            model_type
        )

    # Step 4: Evaluation
    if st.session_state.steps['trained']:
        evaluate_model(
            st.session_state.predictions,
            st.session_state.features,
            model_type,
            theme
        )

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
