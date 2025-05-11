# app.py - Universal ML Platform with 4 Themes
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import datetime

# Configure page
st.set_page_config(
    page_title="ML PRO MADE BY SAMAD KIANI",
    page_icon="https://tse2.mm.bing.net/th?id=OIP.Fkdoyke5qijSDVWyGKJB9QHaHk&pid=Api&P=0&h=220",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configurations
THEMES = {
    "Zombie": {
        "background": "https://png.pngtree.com/background/20230426/original/pngtree-zombie-hand-coming-out-of-the-grave-digital-painting-picture-image_2492185.jpg",
        "primary_color": "#8B0000",
        "secondary_color": "#2F4F4F",
        "text_color": "#FFFFFF",
        "font": "Arial Black",
        "gif": "https://media.giphy.com/media/3o7TKsrf0g3JXxQnY4/giphy.gif"
    },
    "Futuristic": {
        "background": "https://png.pngtree.com/background/20230411/original/pngtree-futuristic-technology-background-with-digital-data-stream-picture-image_2382755.jpg",
        "primary_color": "#00FFFF",
        "secondary_color": "#9400D3",
        "text_color": "#FFFFFF",
        "font": "Courier New",
        "gif": "https://media.giphy.com/media/Ll22OhMLAlVDb8UQWe/giphy.gif"
    },
    "Game of Thrones": {
        "background": "https://png.pngtree.com/background/20230417/original/pngtree-game-of-thrones-dragon-fire-background-picture-image_2436397.jpg",
        "primary_color": "#B22222",
        "secondary_color": "#4682B4",
        "text_color": "#FFFFFF",
        "font": "Times New Roman",
        "gif": "https://media.giphy.com/media/3o7TKz2eMXx8JhnoG4/giphy.gif"
    },
    "Gaming": {
        "background": "https://png.pngtree.com/background/20230408/original/pngtree-neon-gaming-background-with-game-controller-picture-image_2359699.jpg",
        "primary_color": "#FF00FF",
        "secondary_color": "#00FF00",
        "text_color": "#FFFFFF",
        "font": "Press Start 2P",
        "gif": "https://media.giphy.com/media/3o72F7RqPjO6YpsQYM/giphy.gif"
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
        'stock_data': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_theme(theme):
    st.markdown(f"""
    <style>
        body, .stApp {{
            background-image: url('{theme["background"]}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: {theme["text_color"]};
            font-family: {theme["font"]}, sans-serif;
        }}
        .main {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
            color: {theme["text_color"]};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["primary_color"]};
            font-weight: 700;
        }}
        .stButton>button {{
            background-color: {theme["primary_color"]};
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }}
        .stDownloadButton>button {{
            background-color: {theme["secondary_color"]};
            color: white;
            font-weight: bold;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            color: {theme["text_color"]};
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
        }}
        .st-expanderContent {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 1rem;
            border-radius: 10px;
        }}
        .stSelectbox, .stMultiselect, .stSlider, .stTextInput {{
            color: #000000;
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

def show_data_preview(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("### Dataset Preview:")
    st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)

def feature_selection(df):
    with st.expander("🔍 Select Features & Target"):
        st.markdown("<div class='feature-selector'>", unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("Select Target Variable:", numeric_cols, index=len(numeric_cols)-1 if len(numeric_cols) > 0 else 0)
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
                st.success("Features and target confirmed!")
        st.markdown("</div>", unsafe_allow_html=True)

def data_analysis(df, features, target, theme):
    st.header("2. Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Feature-Target Relationships")
        selected_feature = st.selectbox("Select feature to plot:", features)
        fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400)
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': theme["text_color"]}
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
    
    if st.button("🚀 Proceed to Model Training"):
        st.session_state.steps['ready_for_model'] = True

def train_model(features, target, model_type, test_size):
    df = st.session_state.data
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
    
    with st.spinner(f"Training {model_type}..."):
        model.fit(X_train_scaled, y_train)
        st.session_state.model = model
        st.session_state.steps['trained'] = True
        
        y_pred = model.predict(X_test_scaled)
        st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
        st.success("Model trained successfully!")
        st.balloons()

def evaluate_model(predictions, features, model_type, theme):
    st.header("4. Model Evaluation")
    y_test = predictions['y_test']
    y_pred = predictions['y_pred']
    X_test = predictions['X_test']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    with col2:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
    
    st.write("### Actual vs Predicted Values")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
    
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
    
    if model_type == "Random Forest":
        st.write("### Feature Importance")
        importance = pd.DataFrame({'Feature': features, 'Importance': st.session_state.model.feature_importances_})
        importance = importance.sort_values('Importance', ascending=False)
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {'color': theme["text_color"]}
        })
        st.plotly_chart(fig, use_container_width=True)
    
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Predictions", csv, "predictions.csv", "text/csv")

def main():
    setup_session_state()
    
    # Theme selection in sidebar
    with st.sidebar:
        selected_theme = st.selectbox("🎨 Select Theme", list(THEMES.keys()))
        theme = THEMES[selected_theme]
        apply_theme(theme)
    
    # Welcome page with theme GIF
    st.markdown(f'<div class="main">', unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center'><img src='{theme['gif']}' width='300'></div>", unsafe_allow_html=True)
    st.title(f"📊 Universal ML Analysis Platform - {selected_theme} Theme")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_source = st.radio("Data Source:", ["Upload Dataset", "Yahoo Finance"])
        
        if data_source == "Upload Dataset":
            uploaded_file = st.file_uploader("Upload Dataset:", type=["csv", "xlsx"])
        else:
            stock_symbol = st.text_input("Stock Symbol (e.g., AAPL):", "AAPL")
            start_date = st.date_input("Start Date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End Date:", datetime.date.today())
            if st.button("Fetch Stock Data"):
                fetch_stock_data(stock_symbol, start_date, end_date)
        
        st.markdown("---")
        st.header("🧠 Model Settings")
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reset Session", on_click=lambda: st.session_state.clear())

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
            model_type,
            test_size
        )

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        evaluate_model(
            st.session_state.predictions,
            st.session_state.features,
            model_type,
            theme
        )

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()