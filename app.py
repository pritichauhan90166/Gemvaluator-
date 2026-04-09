import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="GemValuator", layout="wide", initial_sidebar_state="collapsed")

# =========================
# CUSTOM CSS (FROM GRADEINSIGHT)
# =========================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg-base:    #1e2130;
    --bg-card:    #252838;
    --bg-sidebar: #1a1d2e;
    --bg-input:   #2d3148;
    --accent:     #e94560;   /* changed to gem red */
    --accent-dim: #c73650;
    --border:     #363a52;
    --border-lt:  #444868;
    --text-hi:    #eef0f8;
    --text-mid:   #a0a8c8;
    --text-lo:    #6b7196;
    --shadow:     0 2px 10px rgba(0,0,0,0.35);
}

/* GLOBAL */
html, body, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
}

h1, h2, h3 {
    color: var(--text-hi) !important;
    font-weight: 700 !important;
}

p, span, div {
    color: var(--text-hi) !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stRadio label {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 6px;
    cursor: pointer;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    border-color: var(--accent);
    background: rgba(233,69,96,0.1);
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    box-shadow: var(--shadow);
}

div[data-testid="stMetricValue"] {
    color: var(--accent);
    font-size: 1.8rem;
}

/* BUTTON */
.stButton > button {
    background: var(--accent);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}

.stButton > button:hover {
    background: var(--accent-dim);
}

/* INPUT */
input, .stTextInput input {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: white;
}

/* TABLE */
.stDataFrame {
    background: var(--bg-card);
    border-radius: 10px;
}

/* HERO HEADER */
.hero {
    background: linear-gradient(135deg, #252838, #2d3148);
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid var(--border);
}

/* RESULT BOX */
.result-box {
    background: linear-gradient(135deg, #2b1a1f, #1e1316);
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    border: 1px solid var(--border);
}

.result-score {
    font-size: 3rem;
    color: var(--accent);
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & DATA
# =========================
@st.cache_resource
def load_model():
    return joblib.load("dt_model.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("Diamonds Prices2022.csv.xlsx")

model = load_model()
df = load_data()

# =========================
# MODEL COMPARISON DATA (YOUR VALUES)
# =========================
results_df = pd.DataFrame({
    "Model": [
        "Decision Tree", "Random Forest", "XGBoost", "LightGBM",
        "Gradient Boosting", "KNN", "Neural Network (MLP)",
        "Linear Regression", "SVR"
    ],
    "MAE": [0.80, 0.78, 12.00, 12.41, 38.06, 47.19, 440.85, 595.93, 1119.19],
    "RMSE": [3.30, 4.76, 18.39, 20.25, 59.05, 329.42, 746.31, 831.92, 2092.97],
    "R2 Score": [1.0000, 1.0000, 1.0000, 1.0000, 0.9996, 0.9885, 0.9411, 0.9268, 0.5367]
})

results_df = results_df.sort_values(by="RMSE")

# =========================
# LOG SYSTEM 
# =========================
if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(message, level="info"):
    import datetime

    time = datetime.datetime.now().strftime("%H:%M:%S")

    if level == "success":
        icon = "✅"
    elif level == "error":
        icon = "❌"
    else:
        icon = "ℹ️"

    log_entry = f"{icon} [{time}] {message}"
    st.session_state.logs.append(log_entry)

    @st.cache_resource
    def load_model():
        model = joblib.load("dt_model.pkl")
        add_log("Model loaded successfully", "success")
        return model

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown('<div style="font-weight:700; font-size:1.1rem;">💎 GEMVALUATOR</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "EDA","Upload & Explore", "Prediction", "Visualization", "Model Comparison","Model Logs"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:0.75rem; color:#a0a8c8; text-align:center;">AI Model · Diamond Price Predictor</p>',
    unsafe_allow_html=True
)


# =========================
# HOME PAGE 
# =========================
if page == "Dashboard":

    # HERO SECTION
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 36px 32px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 28px;
        border: 1px solid #2c2f45;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    ">
        <h1 style="color:#e94560; font-size:2.4rem; margin-bottom:8px;">
            💎 GemValuator
        </h1>
        <p style="color:#c0c6e4; font-size:1rem;">
            AI-Powered Diamond Price Prediction & Analysis Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # KPI METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Diamonds", f"{len(df)}")
    col2.metric("💰 Avg Price", f"{df['price'].mean():.0f}")
    col3.metric("💎 Avg Carat", f"{df['carat'].mean():.2f}")

    st.markdown("---")

    # =========================
    # INFO SECTIONS
    # =========================
    left, right = st.columns(2)

    # LEFT SIDE → ABOUT DIAMONDS
    with left:
        st.markdown("### 💡 Understanding Diamonds")

        st.markdown("""
        Diamond pricing depends on the famous **4Cs**:

        - **Carat** → Weight of the diamond  
        - **Cut** → Quality of shape & shine  
        - **Color** → Whiteness (D = best)  
        - **Clarity** → Internal flaws  

        Higher quality = higher price 💰
        """)

        diamond_table = pd.DataFrame({
            "Factor": ["Carat", "Cut", "Color", "Clarity"],
            "Impact": [
                "Directly increases price",
                "Affects brilliance",
                "Better color = higher value",
                "Fewer flaws = expensive"
            ]
        })

        st.table(diamond_table)

    # RIGHT SIDE → FEATURES USED
    with right:
        st.markdown("### 🔬 Model Features")

        features = [
            ("Carat", "Weight of diamond"),
            ("Cut", "Quality of cut"),
            ("Color", "Diamond color grade"),
            ("Clarity", "Purity level"),
            ("Depth", "Height proportion"),
            ("Table", "Top surface size"),
        ]

        for name, desc in features:
            st.markdown(f"""
            <div style="
                background:#1a1a2e;
                border:1px solid #2c2f45;
                border-radius:10px;
                padding:10px 14px;
                margin-bottom:8px;">
                <span style="color:#e94560; font-weight:600;">{name}</span>
                <span style="color:#c0c6e4; margin-left:10px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # EXTRA INFO SECTION
    # =========================
    col_env, col_safe = st.columns(2)

    with col_env:
        st.markdown("### 📈 Business Impact")
        st.markdown("""
        - Helps jewelers price diamonds accurately  
        - Detects overpriced/underpriced gems  
        - Useful for resale & valuation  
        - Supports inventory decisions  
        """)

    with col_safe:
        st.markdown("### 🤖 Why AI Model?")
        st.markdown("""
        - Removes human bias  
        - Predicts price instantly  
        - Learns from historical data  
        - Improves accuracy over time  
        """)

# =========================
# EDA PAGE
# =========================
elif page == "EDA":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Data Intelligence Dashboard
    </h1>
    """, unsafe_allow_html=True)

    # =========================
    # FILTERS
    # =========================
    st.markdown("### 🎛️ Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_cut = st.multiselect("Cut", df["cut"].unique(), default=df["cut"].unique())

    with col2:
        selected_color = st.multiselect("Color", df["color"].unique(), default=df["color"].unique())

    with col3:
        selected_clarity = st.multiselect("Clarity", df["clarity"].unique(), default=df["clarity"].unique())

    filtered_df = df[
        (df["cut"].isin(selected_cut)) &
        (df["color"].isin(selected_color)) &
        (df["clarity"].isin(selected_clarity))
    ]

    st.markdown("---")

    # =========================
    # KPIs
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💎 Total Diamonds", f"{len(filtered_df):,}")
    c2.metric("💰 Avg Price", f"{filtered_df['price'].mean():,.0f}")
    c3.metric("⚖️ Avg Carat", f"{filtered_df['carat'].mean():.2f}")
    c4.metric("📏 Avg Depth", f"{filtered_df['depth'].mean():.2f}")

    st.markdown("---")

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Relationships"])

    # =========================
    # TAB 1: OVERVIEW
    # =========================
    with tab1:

        col1, col2 = st.columns(2)

        # Cut Distribution
        fig1 = px.bar(filtered_df, x="cut", color="cut", title="Cut Distribution")
        col1.plotly_chart(fig1, use_container_width=True)

        # Color Distribution
        fig2 = px.bar(filtered_df, x="color", color="color", title="Color Distribution")
        col2.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        # Clarity Distribution
        fig3 = px.bar(filtered_df, x="clarity", color="clarity", title="Clarity Distribution")
        col3.plotly_chart(fig3, use_container_width=True)

        # Avg Price by Cut
        avg_price = filtered_df.groupby("cut")["price"].mean().reset_index()
        fig4 = px.bar(avg_price, x="cut", y="price", color="cut", title="Avg Price by Cut")
        col4.plotly_chart(fig4, use_container_width=True)

    # =========================
    # TAB 2: DISTRIBUTIONS
    # =========================
    with tab2:

        feature = st.selectbox("Select Feature", filtered_df.select_dtypes(include=np.number).columns)

        col1, col2 = st.columns(2)

        # Histogram
        fig5 = px.histogram(filtered_df, x=feature, nbins=50, color_discrete_sequence=["#e94560"])
        col1.plotly_chart(fig5, use_container_width=True)

        # Boxplot
        fig6 = px.box(filtered_df, y=feature, color_discrete_sequence=["#0f3460"])
        col2.plotly_chart(fig6, use_container_width=True)

       # =========================
    # TAB 3: RELATIONSHIPS
    # =========================
    with tab3:

        st.subheader("🔗 Correlation Heatmap")

        corr = filtered_df.select_dtypes(include=np.number).corr()

        fig7 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("📈 Scatter Analysis")

        col1, col2 = st.columns(2)

        # Only numeric columns
        numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

        # Smart defaults
        default_x = "carat" if "carat" in numeric_cols else numeric_cols[0]
        default_y = "price" if "price" in numeric_cols else numeric_cols[1]

        x_axis = col1.selectbox("X-axis", numeric_cols, index=numeric_cols.index(default_x))
        y_axis = col2.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(default_y))

        fig8 = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color="cut",
            trendline="ols",
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(fig8, use_container_width=True)

        
# =========================
# PREDICTION PAGE 
# =========================
elif page == "Prediction":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Predict Diamond Price
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#a0a8c8; margin-bottom:20px;">
    Enter diamond features below. The AI model will predict the price instantly.
    </p>
    """, unsafe_allow_html=True)

    # =========================
    # USER GUIDELINES
    # =========================
    st.info("""
    📌 Input Guidelines:
    - Carat: 0.2 – 5  
    - Depth: 55 – 65 %  
    - Table: 50 – 70 %  
    - Dimensions (x, y, z): in millimeters (mm)
    """)

    # =========================
    # FORM INPUT
    # =========================
    with st.form("prediction_form"):

        c1, c2 = st.columns(2)
        carat = c1.number_input("Carat (0.2 – 5.0)", min_value=0.1, max_value=10.0, value=1.0)
        depth = c2.number_input("Depth (%) (55 – 65)", min_value=40.0, max_value=80.0, value=61.0)

        c3, c4 = st.columns(2)
        table = c3.number_input("Table (%) (50 – 70)", min_value=40.0, max_value=90.0, value=57.0)
        x = c4.number_input("Length (x) (mm)", min_value=1.0, max_value=15.0, value=5.0)

        c5, c6 = st.columns(2)
        y = c5.number_input("Width (y) (mm)", min_value=1.0, max_value=15.0, value=5.0)
        z = c6.number_input("Height (z) (mm)", min_value=1.0, max_value=10.0, value=3.0)

        c7, c8, c9 = st.columns(3)
        cut = c7.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = c8.selectbox("Color", ["D","E","F","G","H","I","J"])
        clarity = c9.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

        submitted = st.form_submit_button("⚡ Predict Price")

    # =========================
    # VALIDATION
    # =========================
    if submitted:

        st.write("Button clicked")

        # Basic validation
        if carat <= 0 or x <= 0 or y <= 0 or z <= 0:
            st.error("❌ Carat and dimensions must be greater than 0")
            st.stop()

        if not (50 <= table <= 80):
            st.error("❌ Table should be between 50–80 %")
            st.stop()

        if not (50 <= depth <= 70):
            st.error("❌ Depth should be between 50–70 %")
            st.stop()

        # =========================
        # MODEL INPUT
        # =========================
        
        add_log(f"Prediction started:") 


        try:
                input_data = pd.DataFrame([{
                    "carat": carat,
                    "cut": cut,
                    "color": color,
                    "clarity": clarity,
                    "depth": depth,
                    "table": table,
                    "x": x,
                    "y": y,
                    "z": z
                }])

                df_clean = df.copy()

                if "Unnamed0" in df_clean.columns:
                    df_clean = df_clean.drop("Unnamed0", axis=1)

                combined = pd.concat([df_clean.drop("price", axis=1), input_data], ignore_index=True)
                combined = pd.get_dummies(combined)

                input_processed = combined.tail(1)

                # ✅ align columns
                input_processed = input_processed.reindex(columns=model.feature_names_in_, fill_value=0)

                # ✅ prediction
                prediction = model.predict(input_processed)[0]

                add_log(f"Prediction generated: ₹ {prediction:,.0f}", "success")   # ✅ 2. AFTER SUCCESS

                # ✅ SAVE DATA FOR LOG PAGE
                st.session_state["prediction"] = prediction
                st.session_state["input_data"] = input_data

                # ✅ LOG SYSTEM INIT
                if "logs" not in st.session_state:
                    st.session_state["logs"] = []

                # ✅ ADD LOG ENTRY
                import datetime
                st.session_state.logs.append(
                    f"{datetime.datetime.now().strftime('%H:%M:%S')} → Prediction: ₹ {prediction:,.0f}"
                )

                # =========================
                # CATEGORY + COLORS
                # =========================
                cat_map = [
                    (2000,  "Low Value", "#00FF9D"),
                    (10000, "Moderate", "#FFE600"),
                    (30000, "High Value", "#FF8C00"),
                    (50000, "Premium", "#FF3B30"),
                    (100000, "Luxury", "#3b7dd8"),
                ]

                category, cat_color = "Luxury", "#3b7dd8"
                color_code = "#3b7dd8"  # ✅ DEFAULT FIX

                for threshold, cat, col in cat_map:
                    if prediction <= threshold:
                        category, cat_color = cat, col
                        break

                # ✅ NOW SAFE (outside loop)
                if prediction < 2000:
                    category = "Low Value"
                    color_code = "#00c96e"
                elif prediction < 10000:
                    category = "Medium Value"
                    color_code = "#f5c518"
                elif prediction < 30000:
                    category = "High Value"
                    color_code = "#ff8c00"
                else:
                    category = "Luxury"
                    color_code = "#e94560"
                    color_code = "#e94560"


                # =========================
                # RESULT LAYOUT (OUTSIDE LOOP ✅)
                # =========================
                res_l, res_r = st.columns(2)

                # LEFT
                with res_l:
                    st.markdown(f"""
                <div style="
                    background:#252838;
                    border:1px solid {color_code};
                    border-radius:16px;
                    padding:28px;
                    text-align:center;
                    box-shadow: 0 0 25px {color_code}55;
                ">
                    <p style="color:#a0a8c8; font-size:0.8rem;">Predicted Price</p>
                    <p style="font-size:3rem; color:{color_code}; margin:0;">
                        ₹ {prediction:,.0f}
                    </p>
                    <p style="color:{color_code}; font-weight:600;">
                        {category}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # RIGHT
                with res_r:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        number={
                            'font': {
                                'family': 'Plus Jakarta Sans',
                                'color': cat_color,
                                'size': 28
                            }
                        },
                        gauge={
                            'axis': {
                                'range': [0, 100000],
                                'tickcolor': '#a0a8c8',
                                'tickfont': {'color': '#a0a8c8'}
                            },
                            'bar': {
                                'color': cat_color,
                                'thickness': 0.25
                            },
                            'bgcolor': 'rgba(37,40,56,0.95)',
                            'bordercolor': 'rgba(54,58,82,1)',
                            'steps': [
                                {'range': [0, 2000], 'color': '#00c96e'},
                                {'range': [2000, 10000], 'color': '#f5c518'},
                                {'range': [10000, 30000], 'color': '#ff8c00'},
                                {'range': [30000, 50000], 'color': '#e03131'},
                                {'range': [50000, 100000], 'color': '#7048e8'},
                            ],
                            'threshold': {
                                'line': {'color': cat_color, 'width': 3},
                                'thickness': 0.75,
                                'value': prediction,
                            }
                        }
                    ))

                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans", color="#a0a8c8"),
                        height=260,
                        margin=dict(t=20, b=10, l=20, r=20)
                    )

                    st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
                add_log(f"Prediction failed: {e}", "error")   # ✅ 3. INSIDE EXCEPT

                st.error(f"Prediction error: {e}")

# =========================
#  Upload & Explore Data 
# =========================
elif page == "Upload & Explore":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Upload & Explore Data
    </h1>
    """, unsafe_allow_html=True)

    # SESSION STATE
    if "bulk_df" not in st.session_state:
        st.session_state.bulk_df = None

    file = st.file_uploader(
        "Upload File",
        type=["csv", "xlsx", "json"],
        key="bulk_file"
    )

    drive_link = st.text_input("Or paste Google Drive file link and press Enter")

    # =========================
    # SAMPLE DATA BUTTON
    # =========================
    if st.button("Load Sample Dataset"):
        try:
            sample_df = pd.read_excel("Book1.xlsx")
            sample_df = sample_df.dropna().sample(200)

            st.session_state.bulk_df = sample_df
            st.success("Sample dataset loaded successfully")

        except Exception as e:
            st.error(f"Error loading sample data: {e}")

    # =========================
    # DRIVE LINK FUNCTION
    # =========================
    def convert_drive_link(link):
        import re
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", link)

        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"

        return None

    # =========================
    # FILE LOAD
    # =========================
    if file:
        file_type = file.name.split(".")[-1].lower()

        if file_type == "csv":
            st.session_state.bulk_df = pd.read_csv(file)

        elif file_type == "xlsx":
            st.session_state.bulk_df = pd.read_excel(file)

        elif file_type == "json":
            df_json = pd.read_json(file)
            if isinstance(df_json.iloc[0], dict):
                df_json = pd.json_normalize(df_json)
            st.session_state.bulk_df = df_json

        st.success(f"{file_type.upper()} file loaded successfully")

    elif drive_link:
        download_link = convert_drive_link(drive_link)

        if download_link:
            try:
                import requests
                from io import BytesIO

                response = requests.get(download_link)
                file_bytes = BytesIO(response.content)

                try:
                    st.session_state.bulk_df = pd.read_csv(file_bytes)
                except:
                    file_bytes.seek(0)
                    try:
                        st.session_state.bulk_df = pd.read_excel(file_bytes)
                    except:
                        file_bytes.seek(0)
                        df_json = pd.read_json(file_bytes)
                        if isinstance(df_json.iloc[0], dict):
                            df_json = pd.json_normalize(df_json)
                        st.session_state.bulk_df = df_json

                st.success("File loaded from Google Drive")

            except Exception as e:
                st.error(f"Drive load failed: {e}")

        else:
            st.error("Invalid Google Drive link")

    # =========================
    # MAIN LOGIC
    # =========================
    bulk_df = st.session_state.bulk_df

    if bulk_df is None:
        st.info("Upload a file, load sample data, or use Google Drive link.")

    else:
        bulk_df = bulk_df.replace(r'^\s*$', np.nan, regex=True)
        bulk_df = bulk_df.dropna()

        st.subheader("📊 Uploaded Data")
        st.dataframe(bulk_df.head())

        # =========================
        # FILTERING
        # =========================
        col = st.sidebar.selectbox("Filter Column (Bulk)", bulk_df.columns)

        if pd.api.types.is_numeric_dtype(bulk_df[col]):
            min_val = float(bulk_df[col].min())
            max_val = float(bulk_df[col].max())

            val = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
            filtered_df = bulk_df[
                (bulk_df[col] >= val[0]) & (bulk_df[col] <= val[1])
            ]
        else:
            val = st.sidebar.selectbox("Value", bulk_df[col].unique())
            filtered_df = bulk_df[bulk_df[col] == val]

        st.subheader("Filtered Data")
        st.dataframe(filtered_df)

        # =========================
        # PREDICTION
        # =========================
        try:
            df_clean = df.copy()

            if "Unnamed0" in df_clean.columns:
                df_clean = df_clean.drop("Unnamed0", axis=1)

            combined = pd.concat(
                [df_clean.drop("price", axis=1), filtered_df],
                ignore_index=True
            )

            combined = pd.get_dummies(combined)
            bulk_processed = combined.tail(len(filtered_df))

            # align columns
            for col in model.feature_names_in_:
                if col not in bulk_processed.columns:
                    bulk_processed[col] = 0

            bulk_processed = bulk_processed[model.feature_names_in_]

            predictions = model.predict(bulk_processed)

            filtered_df["Predicted Price"] = predictions

            st.success("✅ Predictions generated!")

            st.dataframe(filtered_df)

        except Exception as e:
            st.error(f"Prediction error: {e}")

        # =========================
        # DOWNLOAD RESULTS
        # =========================
        csv = filtered_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="bulk_predictions.csv",
            mime="text/csv"
        )

        # =========================
        # VISUALIZATION
        # =========================
        num_cols = filtered_df.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            c = st.selectbox("Histogram Column", num_cols)
            fig = px.histogram(filtered_df, x=c)
            st.plotly_chart(fig)

        if len(num_cols) >= 2:
            x = st.selectbox("X Axis", num_cols)
            y = st.selectbox("Y Axis", [i for i in num_cols if i != x])

            fig2 = px.scatter(filtered_df, x=x, y=y)
            st.plotly_chart(fig2)

# =========================
# VISUALIZATION PAGE
# =========================
elif page == "Visualization":

    st.title("Data Insights")

    st.write(df.head())
    st.write(df.describe())

    # Price Distribution
    st.title("HistPlot")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    # Scatter
    st.title("ScatterPlot")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["carat"], y=df["price"], ax=ax2)
    st.pyplot(fig2)

    # Heatmap
    st.title("HeatMap")

    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=["number"]).corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # Plotly
    st.title("BoxPlot")

    fig4 = px.box(df, x="cut", y="price", color="cut")
    st.plotly_chart(fig4)

    # =========================
    # Feature Importance (FIXED ✅)
    # =========================
    st.subheader("Feature Importance")

    try:
        importance = model.feature_importances_
        feature_names = model.feature_names_in_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.write(importance_df)
        st.title("BarPlot")


        fig5, ax5 = plt.subplots()
        sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", ax=ax5)
        st.pyplot(fig5)

    except Exception as e:
        st.error(f"Feature importance error: {e}")

# =========================
# MODEL COMPARISON (NEW 🔥)
# =========================
elif page == "Model Comparison":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Model Comparison
    </h1>
    """, unsafe_allow_html=True)

    # RMSE Graph
    st.subheader("RMSE Comparison")
    fig = px.bar(results_df, x="Model", y="RMSE", color="Model", text="RMSE")
    st.plotly_chart(fig, use_container_width=True)

    # MAE Graph
    st.subheader("MAE Comparison")
    fig = px.bar(results_df, x="Model", y="MAE", color="Model", text="MAE")
    st.plotly_chart(fig, use_container_width=True)

    # R2 Score Graph
    st.subheader("R2 Score Comparison")
    fig = px.bar(results_df, x="Model", y="R2 Score", color="Model", text="R2 Score")
    st.plotly_chart(fig, use_container_width=True)

    # All metrics
    st.subheader("All Metrics Comparison")
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")

    fig2 = px.bar(melted, x="Model", y="Value", color="Metric", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    # Table
    st.dataframe(results_df)

    # Best model
    best = results_df.iloc[0]

    st.success(f"""
🏆 Best Model: {best['Model']}

RMSE: {best['RMSE']}
MAE: {best['MAE']}
R² Score: {best['R2 Score']}
""")

    # ✅ FIXED POSITION
    st.info(
        "The Decision Tree model outperforms others by minimizing prediction errors (lowest RMSE & MAE) while maximizing explained variance (highest R²), making it the most reliable model for this dataset."
    )
# =========================
# MODEL LOG PAGE
# =========================
elif page == "Model Logs":

    st.markdown("""<h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Model Logs</h1>""", unsafe_allow_html=True)

    # =========================
    # MODEL INFO BANNER
    # =========================
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(233,69,96,0.06),rgba(233,69,96,0.04));
                border:1px solid #363a52; border-radius:14px;
                padding:16px 20px; margin-bottom:20px;">
        <span style="font-family:'Plus Jakarta Sans'; color:#e94560; font-size:0.95rem; font-weight:600;">
            Final Model: Decision Tree Regressor
        </span>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "1.00")
    col2.metric("RMSE", "3.30")
    col3.metric("MAE", "0.80")

    st.markdown("---")

    left_l, right_l = st.columns(2, gap="large")

    # =========================
    # LEFT SIDE
    # =========================
    with left_l:
        st.markdown("#### Features Used")

        features_used = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
        st.markdown(" · ".join(features_used))

        st.markdown("#### Pipeline")

        steps = ["Data Cleaning", "Encoding", "Training", "Prediction"]
        st.markdown(" → ".join(
            [f'<span style="color:#e94560;font-family:Plus Jakarta Sans;'
             f'font-size:0.78rem;font-weight:600;">{s}</span>' for s in steps]
        ), unsafe_allow_html=True)

        st.markdown("#### Dataset Info")
        st.markdown(f"""
        Shape: {df.shape[0]} rows × {df.shape[1]} columns  
        Missing Values: {df.isnull().sum().sum()}
        """)

    # =========================
    # RIGHT SIDE
    # =========================
    with right_l:
        st.markdown("#### Last Prediction")

        if "prediction" in st.session_state:
            st.markdown(f"""
            <div style="
                background:#252838;
                border:1px solid #e94560;
                border-radius:12px;
                padding:20px;
                text-align:center;
                box-shadow: 0 0 15px rgba(233,69,96,0.4);
            ">
                <p style="color:#a0a8c8; font-size:0.8rem;">Predicted Price</p>
                <p style="font-size:2.5rem; color:#e94560; margin:0;">
                    ₹ {st.session_state['prediction']:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No prediction yet")

        st.markdown("#### Last Input")

        if "input_data" in st.session_state:
            st.dataframe(st.session_state["input_data"])
        else:
            st.warning("No input data yet")

    # =========================
    # EVENT LOGS (AQI STYLE)
    # =========================
    st.markdown("#### Event Logs")

    if st.session_state.logs:

        log_html = "".join([
            f'<div style="font-family:monospace; font-size:0.82rem;'
            f'color:#a0a8c8; padding:5px 10px; border-left:2px solid rgba(233,69,96,0.25);'
            f'margin-bottom:4px;">{log}</div>'
            for log in reversed(st.session_state.logs)
        ])

        st.markdown(
            f'<div style="background:#1e2130; border:1px solid #363a52;'
            f'border-radius:10px; padding:12px; max-height:280px; overflow-y:auto;">'
            f'{log_html}</div>',
            unsafe_allow_html=True
        )

        # BUTTONS
        _, c1, c2, _ = st.columns([1, 2, 2, 1])

        with c1:
            st.download_button(
                "Download Logs",
                data="\n".join(st.session_state.logs),
                file_name="gemvaluator_logs.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with c2:
            if st.button("Clear Logs", use_container_width=True):
                st.session_state.logs = []
                st.rerun()

    else:
        st.info("No logs yet. Run a prediction to see activity here.")