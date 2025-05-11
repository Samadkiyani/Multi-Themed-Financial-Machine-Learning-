# app.py - Universal ML Platform with 4 New Themes
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import yfinance as yf
import datetime
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="ML PRO MADE BY SAMAD KIANI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configurations
THEMES = {
    "Avengers": {
        "background": "https://images.unsplash.com/photo-1633613286848-e6f43bbafb8d?ixlib=rb-4.0.3",
        "primary_color": "#ED1D24",  # Marvel Red
        "secondary_color": "#0057B7",  # Marvel Blue
        "text_color": "#FFFFFF",
        "font": "Arial Black",
        "gif": "https://media.giphy.com/media/l3vRno1vowh3V8G3C/giphy.gif",
        "button_style": "marvel",
        "model": "Linear Regression"
    },
    "Zombie": {
        "background": "https://images.unsplash.com/photo-1506815444479-bfdb1e96c566?ixlib=rb-4.0.3",
        "primary_color": "#8B0000",  # Blood Red
        "secondary_color": "#2F4F4F",  # Dark Slate Gray
        "text_color": "#FFFFFF",
        "font": "Chiller",
        "gif": "https://media.giphy.com/media/3o7TKsrf0g3JXxQnY4/giphy.gif",
        "button_style": "horror",
        "model": "Logistic Regression"
    },
    "Game of Thrones": {
        "background": "https://images.unsplash.com/photo-1534447677768-be436bb09401?ixlib=rb-4.0.3",
        "primary_color": "#B22222",  # Firebrick Red
        "secondary_color": "#4682B4",  # Steel Blue
        "text_color": "#FFFFFF",
        "font": "Game of Thrones",
        "gif": "https://media.giphy.com/media/3o7TKz2eMXx8JhnoG4/giphy.gif",
        "button_style": "medieval",
        "model": "K-Means Clustering"
    },
    "PUBG": {
        "background": "https://images.unsplash.com/photo-1542751371-adc38448a05e?ixlib=rb-4.0.3",
        "primary_color": "#FFA500",  # Orange
        "secondary_color": "#556B2F",  # Dark Olive Green
        "text_color": "#FFFFFF",
        "font": "Impact",
        "gif": "https://media.giphy.com/media/3o72F7RqPjO6YpsQYM/giphy.gif",
        "button_style": "military",
        "model": "Random Forest"
    }
}

def setup_session_state():
    session_defaults = {
        'data': None, 
        'model': None, 
        'features': [], 
        'target': None,
        'steps': {'loaded': False, 'processed': False, 'trained': False, 'ready_for_model': False},
        'predictions': None, 
        'stock_data': None,
        'cluster_labels': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_theme(theme):
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family={theme["font"].replace(" ", "+")}&display=swap');
        
        body, .stApp {{
            background-image: url('{theme["background"]}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: {theme["text_color"]};
            font-family: '{theme["font"]}', sans-serif;
        }}
        .main {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
            color: {theme["text_color"]};
            backdrop-filter: blur(5px);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["primary_color"]};
            font-weight: 700;
            text-shadow: 2px 2px 4px #000000;
        }}
        .stButton>button {{
            background-color: {theme["primary_color"]};
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            border: none;
            font-family: '{theme["font"]}', sans-serif;
        }}
        .stButton>button:hover {{
            background-color: {theme["secondary_color"]};
            color: white;
        }}
        .stDownloadButton>button {{
            background-color: {theme["secondary_color"]};
            color: white;
            font-weight: bold;
            font-family: '{theme["font"]}', sans-serif;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            color: {theme["text_color"]};
            border-left: 4px solid {theme["primary_color"]};
        }}
        .data-warning {{
            color: #FF0000;
            font-weight: bold;
        }}
        .feature-selector {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 15px;
            border-radius: 10px;
            color: {theme["text_color"]};
            border: 1px solid {theme["secondary_color"]};
        }}
        .st-expanderContent {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 1rem;
            border-radius: 10px;
        }}
        .stSelectbox, .stMultiselect, .stSlider, .stTextInput {{
            color: #000000;
        }}
        .st-bb {{
            background-color: transparent;
        }}
        .st-at {{
            background-color: {theme["primary_color"]};
        }}
        div[data-baseweb="select"] > div {{
            background-color: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Dataset needs at least 2 numeric columns for analysis")
            return None
            
        st.session_state.data = df
        st.session_state.steps['loaded'] = True
        st.success(f"✅ Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def fetch_stock_data(stock_symbol, start_date, end_date):
    with st.spinner(f"Fetching {stock_symbol} data..."):
        try:
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
            if not stock_data.empty:
                st.session_state.data = stock_data.reset_index()
                st.session_state.steps['loaded'] = True
                st.success(f"✅ Successfully loaded {len(stock_data)} records")
                return stock_data
            else:
                st.error("No data found for this stock symbol")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

def show_data_preview(df, theme):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("### Dataset Preview:")
    st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols).set_properties(**{
        'background-color': 'rgba(0, 0, 0, 0.7)',
        'color': theme["text_color"]
    }), height=250)

def feature_selection(df, theme):
    with st.expander("🔍 Select Features & Target", expanded=True):
        st.markdown("<div class='feature-selector'>", unsafe_allow_html=True)
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # For classification problems, check if target is categorical
        is_classification = False
        if len(numeric_cols) > 0:
            target = st.selectbox("Select Target Variable:", all_cols, index=len(all_cols)-1)
            
            # If target is not numeric, encode it for classification
            if target not in numeric_cols:
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target])
                is_classification = True
                st.info(f"Non-numeric target detected. Converted to numeric values for {theme['model']}.")
            
            default_features = [col for col in numeric_cols if col != target][:3]
            features = st.multiselect("Select Features:", numeric_cols, default=default_features)
            
            if st.button("Confirm Selection"):
                if len(features) < 1:
                    st.error("Please select at least one feature")
                elif target in features:
                    st.error("Target variable cannot be a feature")
                else:
                    st.session_state.features = features
                    st.session_state.target = target
                    st.session_state.steps['processed'] = True
                    st.session_state.is_classification = is_classification
                    st.success("Features and target confirmed!")
        else:
            st.error("No numeric columns found for analysis")
        st.markdown("</div>", unsafe_allow_html=True)

def data_analysis(df, features, target, theme):
    st.header("🔬 Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### {theme['model']} Preparation")
        if theme['model'] in ["Logistic Regression", "K-Means Clustering"]:
            st.write(f"Preparing data for {theme['model']}...")
            if st.session_state.is_classification:
                st.write("Classification problem detected")
            else:
                st.write("Using numeric features for clustering")
        
        st.write("### Feature-Target Relationships")
        selected_feature = st.selectbox("Select feature to plot:", features)
        
        if theme['model'] == "K-Means Clustering":
            fig = px.scatter(df, x=selected_feature, y=target, height=400)
        else:
            fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400)
        
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': theme["text_color"]},
            'xaxis': {'gridcolor': 'rgba(255, 255, 255, 0.1)'},
            'yaxis': {'gridcolor': 'rgba(255, 255, 255, 0.1)'}
        })
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.write("### Correlation Matrix")
        corr_matrix = df[features + [target]].corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Blues', aspect="auto")
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': theme["text_color"]}
        })
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button(f"🚀 Train {theme['model']} Model", key="train_button"):
        st.session_state.steps['ready_for_model'] = True

def train_model(features, target, model_type, test_size, theme):
    df = st.session_state.data
    X = df[features]
    y = df[target]
    
    if model_type == "K-Means Clustering":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with st.spinner(f"Training {model_type}..."):
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            st.session_state.model = kmeans
            st.session_state.cluster_labels = clusters
            st.session_state.steps['trained'] = True
            st.session_state.X = X
            st.session_state.y = y
            
            # Calculate silhouette score
            score = silhouette_score(X_scaled, clusters)
            st.session_state.silhouette_score = score
            
            st.success(f"Model trained successfully! Silhouette Score: {score:.2f}")
            st.balloons()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(f"Training {model_type}..."):
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            
            if model_type == "Logistic Regression":
                accuracy = accuracy_score(y_test, y_pred.round())
                st.session_state.accuracy = accuracy
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.session_state.rmse = rmse
                st.session_state.r2 = r2
                st.success(f"Model trained successfully! RMSE: {rmse:.2f}, R²: {r2:.2f}")
            st.balloons()

def evaluate_model(predictions, features, model_type, theme):
    st.header("📊 Model Evaluation")
    
    if model_type == "K-Means Clustering":
        st.write("### Cluster Visualization")
        X = st.session_state.X
        clusters = st.session_state.cluster_labels
        
        # 3D plot if we have at least 3 features
        if len(features) >= 3:
            fig = px.scatter_3d(
                x=X[features[0]],
                y=X[features[1]],
                z=X[features[2]],
                color=clusters,
                title=f"{model_type} Clusters",
                labels={'color': 'Cluster'}
            )
        else:
            # 2D plot if fewer features
            fig = px.scatter(
                x=X[features[0]],
                y=X[features[1]] if len(features) > 1 else X[features[0]],
                color=clusters,
                title=f"{model_type} Clusters",
                labels={'color': 'Cluster'}
            )
        
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': theme["text_color"]}
        })
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Silhouette Score", f"{st.session_state.silhouette_score:.2f}")
        
        # Cluster statistics
        st.write("### Cluster Statistics")
        cluster_df = pd.DataFrame({
            'Cluster': clusters,
            'Count': np.ones(len(clusters))
        })
        st.bar_chart(cluster_df.groupby('Cluster').count())
        
    else:
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            if model_type == "Logistic Regression":
                st.metric("Accuracy", f"{st.session_state.accuracy:.2f}")
            else:
                st.metric("RMSE", f"{st.session_state.rmse:.2f}")
        with col2:
            if model_type != "Logistic Regression":
                st.metric("R² Score", f"{st.session_state.r2:.2f}")
        
        st.write("### Actual vs Predicted Values")
        results = pd.DataFrame({
            'Actual': y_test, 
            'Predicted': y_pred.round() if model_type == "Logistic Regression" else y_pred
        }).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Actual'], 
            name='Actual', 
            mode='markers', 
            marker=dict(color=theme["primary_color"])
        ))
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Predicted'], 
            name='Predicted', 
            mode='markers', 
            marker=dict(color=theme["secondary_color"])
        ))
        fig.update_layout({
            'xaxis_title': "Sample Index",
            'yaxis_title': "Value",
            'height': 500,
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': theme["text_color"]}
        })
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type in ["Random Forest", "Linear Regression"]:
            st.write("### Feature Importance")
            if model_type == "Random Forest":
                importance = pd.DataFrame({
                    'Feature': features, 
                    'Importance': st.session_state.model.feature_importances_
                })
            else:  # Linear Regression
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': np.abs(st.session_state.model.coef_[0])
                })
            
            importance = importance.sort_values('Importance', ascending=False)
            fig = px.bar(
                importance, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                color='Importance', 
                color_continuous_scale=[theme["primary_color"], theme["secondary_color"]]
            )
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': {'color': theme["text_color"]}
            })
            st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    if model_type != "K-Means Clustering":
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Predictions", 
            csv, 
            "predictions.csv", 
            "text/csv",
            key='download_results'
        )
    else:
        cluster_results = pd.DataFrame({
            'Features': st.session_state.X.values.tolist(),
            'Cluster': st.session_state.cluster_labels
        })
        csv = cluster_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Cluster Assignments", 
            csv, 
            "clusters.csv", 
            "text/csv",
            key='download_clusters'
        )

def welcome_page(theme):
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="{theme['gif']}" width="400">
        <h1 style="color: {theme['primary_color']};">{theme['model']} Platform - {theme['font']} Theme</h1>
        <h3 style="color: {theme['secondary_color']};">Analyze Financial Data with Machine Learning</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-selector">
            <h3 style="color: {theme['primary_color']};">📊 Key Features</h3>
            <ul>
                <li>Upload your own financial datasets</li>
                <li>Fetch real-time stock market data</li>
                <li>Advanced {theme['model']} analysis</li>
                <li>Interactive visualizations</li>
                <li>Export results for further analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-selector">
            <h3 style="color: {theme['primary_color']};">⚙️ How It Works</h3>
            <ol>
                <li>Select your data source</li>
                <li>Choose features and target variable</li>
                <li>Explore data with interactive charts</li>
                <li>Train the {theme['model']} model</li>
                <li>Evaluate and interpret results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def main():
    setup_session_state()
    
    # Theme selection in sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🎨 THEME SELECTOR</h1>", unsafe_allow_html=True)
        selected_theme = st.selectbox("", list(THEMES.keys()), label_visibility="collapsed")
        theme = THEMES[selected_theme]
        apply_theme(theme)
    
    # Main content area
    st.markdown(f'<div class="main">', unsafe_allow_html=True)
    
    # Welcome page
    if not st.session_state.steps['loaded']:
        welcome_page(theme)
    else:
        st.title(f"{theme['model']} Analysis - {selected_theme} Theme")
        st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Upload Dataset", "Yahoo Finance"])
        
        if data_source == "Upload Dataset":
            uploaded_file = st.file_uploader("Upload Dataset:", type=["csv", "xlsx"])
            if uploaded_file:
                load_data(uploaded_file)
        else:
            stock_symbol = st.text_input("Stock Symbol (e.g., AAPL):", "AAPL")
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
            if st.button("Fetch Stock Data"):
                fetch_stock_data(stock_symbol, start_date, end_date)
        
        st.markdown("---")
        st.header("🧠 Model Settings")
        st.write(f"Selected Model: **{theme['model']}**")
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    if st.session_state.steps['loaded']:
        st.header("📂 Data Upload & Selection")
        show_data_preview(st.session_state.data, theme)
        feature_selection(st.session_state.data, theme)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        data_analysis(
            st.session_state.data,
            st.session_state.features,
            st.session_state.target,
            theme
        )

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        train_model(
            st.session_state.features,
            st.session_state.target,
            theme['model'],
            test_size,
            theme
        )

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        if theme['model'] == "K-Means Clustering":
            evaluate_model(
                None,
                st.session_state.features,
                theme['model'],
                theme
            )
        else:
            evaluate_model(
                st.session_state.predictions,
                st.session_state.features,
                theme['model'],
                theme
            )

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
