# app.py - Universal ML Platform for Any Dataset
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openpyxl
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure page
st.set_page_config(
    page_title="ML PRO MADE BY SAMAD KIANI",
    page_icon="https://tse2.mm.bing.net/th?id=OIP.Fkdoyke5qijSDVWyGKJB9QHaHk&pid=Api&P=0&h=220",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configurations
THEMES = {
    "Zombie Apocalypse": {
        "background": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHB6YTZzdjM3anpzOTdtdXg5NXBpM2VlbDNmNzlvenBmMnpicXJ6eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/13bD1lqmU6fRbq/giphy.gif",
        "primary_color": "#ff0000",
        "header_color": "#ffffff",
        "text_color": "#000000",
        "sidebar_bg": "rgba(255, 255, 255, 0.9)",
        "sidebar_text": "#000000",
        "font_family": "'Creepster', cursive"
    },
    "Futuristic Voyage": {
        "background": "linear-gradient(to right, #00f3ff, #00d2ff, #00b7ff, #00a0ff, #008aff, #0075ff, #0060ff)",
        "primary_color": "#00f3ff",
        "header_color": "#000000",
        "text_color": "#ffffff",
        "sidebar_bg": "rgba(255, 255, 255, 0.8)",
        "sidebar_text": "#000000",
        "font_family": "'Orbitron', sans-serif"
    },
    "Game of Kingdoms": {
        "background": "https://media.giphy.com/media/3o7TKwmnDgQb5jemjK/giphy.gif",
        "primary_color": "#ffcc00",
        "header_color": "#000000",
        "text_color": "#ffffff",
        "sidebar_bg": "rgba(255, 255, 255, 0.8)",
        "sidebar_text": "#000000",
        "font_family": "'Cinzel', serif"
    },
    "Pixel Playground": {
        "background": "linear-gradient(to right, #ff6b6b, #ffa500, #ffff00, #00ff00, #00ffff, #0000ff, #9b59b6)",
        "primary_color": "#39ff14",
        "header_color": "#000000",
        "text_color": "#ffffff",
        "sidebar_bg": "rgba(255, 255, 255, 0.9)",
        "sidebar_text": "#000000",
        "font_family": "'Press Start 2P', cursive"
    }
}

def apply_theme(theme):
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        body, .stApp {{
            background-image: url('{theme["background"]}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: {theme["text_color"]};
            font-family: '{theme["font_family"]}', cursive;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            color: {theme["text_color"]};
        }}
        h1, h2 {{
            color: {theme["header_color"]};
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
            background-color: #28a745;
            color: white;
            font-weight: bold;
        }}
        .sidebar .sidebar-content {{
            background-color: {theme["sidebar_bg"]};
            color: {theme["sidebar_text"]};
            padding: 20px;
            border-radius: 10px;
        }}
        .data-warning {{
            color: #c0392b;
            font-weight: bold;
        }}
        .feature-selector {{
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 10px;
            color: #333;
        }}
        .st-expanderContent {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 1rem;
            border-radius: 10px;
        }}
        @keyframes fly {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(100vw); }}
        }}
        .airplane {{
            position: absolute;
            top: 20%;
            left: -10%;
            animation: fly 5s linear infinite;
        }}
    </style>
    """, unsafe_allow_html=True)

    if theme["background"].endswith(".gif"):
        st.markdown("""
        <script>
            setInterval(function() {{
                var airplane = document.querySelector('.airplane');
                airplane.style.left = '100vw';
            }}, 5000);
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <img src="https://cdn-icons-png.flaticon.com/512/619/619167.png" class="airplane" width="50" />
        """, unsafe_allow_html=True)

# Main Function
def main():
    st.sidebar.header("⚙️ Configuration")
    theme_name = st.sidebar.selectbox("Select Theme:", list(THEMES.keys()))
    theme = THEMES[theme_name]
    apply_theme(theme)

    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("📊 Universal ML Analysis Platform")
    st.markdown("---")
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'features': [], 'target': None,
        'steps': {'loaded': False, 'processed': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    uploaded_file = st.sidebar.file_uploader("Upload Dataset:", type=["csv", "xlsx"])
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Model Settings")
    model_type = st.sidebar.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
    test_size = st.sidebar.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
    st.sidebar.button("Reset Session", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    st.header("1. Data Upload & Selection")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("Dataset needs at least 2 numeric columns for analysis")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            st.write("### Dataset Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander("🔍 Select Features & Target"):
                st.markdown("<div class='feature-selector'>", unsafe_allow_html=True)
                all_cols = df.columns.tolist()
                target = st.selectbox("Select Target Variable:", numeric_cols, index=len(numeric_cols)-1)
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
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.markdown("""
        <div class='feature-selector'>
        📁 **How to Use:**
        1. Upload any CSV or Excel file with numeric data  
        2. Select target variable (what you want to predict)  
        3. Choose features (variables used for prediction)  
        4. The system will automatically handle the rest  
        </div>
        """, unsafe_allow_html=True)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        st.header("2. Data Analysis")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Feature-Target Relationships")
            selected_feature = st.selectbox("Select feature to plot:", features)
            fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400, color=selected_feature, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### Correlation Matrix")
            corr_matrix = df[features + [target]].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Inferno', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 Proceed to Model Training"):
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header("3. Model Training")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
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

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.header("4. Model Evaluation")
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}", delta_color="inverse")
        with col2:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}", delta_color="inverse")
        
        st.write("### Actual vs Predicted Values")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', mode='markers', marker=dict(color='#2a4a7c')))
        fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'], name='Predicted', mode='markers', marker=dict(color='#4CAF50')))
        fig.update_layout(xaxis_title="Sample Index", yaxis_title="Value", height=500, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.write("### Feature Importance")
            importance = pd.DataFrame({'Feature': st.session_state.features, 'Importance': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Inferno')
            st.plotly_chart(fig, use_container_width=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Predictions", csv, "predictions.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
