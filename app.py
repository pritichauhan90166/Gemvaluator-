import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="GemValuator", layout="wide", initial_sidebar_state="collapsed")

# =========================
# CUSTOM CSS (FROM GRADEINSIGHT)
# =========================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 2rem; border-radius: 12px; text-align: center;
}
.main-header h1 { color: #e94560; }
.metric-card {
    background: #1a1a2e; border-radius: 10px;
    padding: 1rem; text-align: center;
}
.result-box {
    background: linear-gradient(135deg, #0d1f12, #081a2b);
    border-radius: 20px; padding: 2rem; text-align: center;
}
.result-score { font-size: 3rem; font-weight: bold; }
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
# SIDEBAR
# =========================
with st.sidebar:
    st.title("💎 GemValuator")
    page = st.radio("", ["Dashboard", "EDA", "Prediction", "Visualization", "Model Comparison","Model Logs"], label_visibility="collapsed")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
<h1>💎 GemValuator</h1>
<p>AI-powered diamond price prediction</p>
</div>
""", unsafe_allow_html=True)


# =========================
# DASHBOARD
# =========================
if page == "Dashboard":

    # KPI CARDS
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='metric-card'><h2>{len(df)}</h2><p>Total Diamonds</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>{df['price'].mean():.0f}</h2><p>Avg Price</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h2>{df['carat'].mean():.2f}</h2><p>Avg Carat</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ROW 1
    col1, col2 = st.columns(2)

    cut_counts = df['cut'].value_counts().reset_index()
    cut_counts.columns = ['cut', 'count']
    fig_cut = px.bar(cut_counts, x='cut', y='count', text='count', title="Cut Distribution")
    fig_cut.update_layout(template="plotly_dark")

    color_counts = df['color'].value_counts().reset_index()
    color_counts.columns = ['color', 'count']
    fig_color = px.bar(color_counts, x='color', y='count', text='count', title="Color Distribution")
    fig_color.update_layout(template="plotly_dark")

    col1.plotly_chart(fig_cut, use_container_width=True)
    col2.plotly_chart(fig_color, use_container_width=True)

    # ROW 2
    col3, col4 = st.columns(2)

    clarity_counts = df['clarity'].value_counts().reset_index()
    clarity_counts.columns = ['clarity', 'count']
    fig_clarity = px.bar(clarity_counts, x='clarity', y='count', text='count', title="Clarity Distribution")
    fig_clarity.update_layout(template="plotly_dark")

    avg_price_cut = df.groupby('cut')['price'].mean().reset_index()
    fig_avg_cut = px.bar(avg_price_cut, x='cut', y='price', text_auto='.2s', title="Avg Price by Cut")
    fig_avg_cut.update_layout(template="plotly_dark")

    col3.plotly_chart(fig_clarity, use_container_width=True)
    col4.plotly_chart(fig_avg_cut, use_container_width=True)

    # =========================
# EDA PAGE
# =========================
elif page == "EDA":

    st.markdown("##Data Intelligence Dashboard")

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

        x_axis = col1.selectbox("X-axis", filtered_df.columns, index=0)
        y_axis = col2.selectbox("Y-axis", filtered_df.columns, index=6)

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

    st.title("Diamond Price Prediction")

    carat = st.slider("Carat", 0.0, 5.0, 1.0, 0.01)
    depth = st.slider("Depth", 40.0, 80.0, 60.0, 0.1)
    table = st.slider("Table", 40.0, 100.0, 55.0, 0.1)
    x = st.slider("Length (x)", 0.0, 15.0, 5.0, 0.01)
    y = st.slider("Width (y)", 0.0, 15.0, 5.0, 0.01)
    z = st.slider("Height (z)", 0.0, 10.0, 3.0, 0.01)

    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
    clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

    # =========================
    # CREATE INPUT DATA (RAW)
    # =========================
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

    # =========================
    # APPLY SAME PREPROCESSING
    # =========================
    df_clean = df.copy()

    if "Unnamed0" in df_clean.columns:
        df_clean = df_clean.drop("Unnamed0", axis=1)

    combined = pd.concat([df_clean.drop("price", axis=1), input_data], ignore_index=True)
    combined = pd.get_dummies(combined)

    input_processed = combined.tail(1)

    missing_cols = set(model.feature_names_in_) - set(input_processed.columns)
    for col in missing_cols:
        input_processed[col] = 0

    input_processed = input_processed[model.feature_names_in_]

    # =========================
    # DISPLAY
    # =========================
    st.write("### Input Data")
    st.write(input_data)

    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_processed)

            st.session_state["input_data"] = input_data
            st.session_state["prediction"] = prediction[0]

            if "history" not in st.session_state:
                st.session_state["history"] = []

            st.session_state["history"].append({
                "prediction": prediction[0]
            })

            st.success(f"💰 Predicted Price: {prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

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

    st.title("📊 Model Comparison")

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

    st.title("Model Logs")

    # Model Info
    st.subheader("Model Info")
    st.write({
        "Model": "Decision Tree",
        "Version": "1.0"
    })

    # Dataset Info
    st.subheader("Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Performance (edit if you have real values)
    st.subheader("Performance")
    st.metric("RMSE", 3.3)
    st.metric("R² Score", 1.0)
    st.metric("MAE", 0.8)


    # Last Prediction
    st.subheader("Last Prediction")

    if "prediction" in st.session_state:
        st.success(f"💰 {st.session_state['prediction']:.2f}")
    else:
        st.warning("No prediction yet")

    # Input Log
    if "input_data" in st.session_state:
        st.write(st.session_state["input_data"])

    # History
    st.subheader("Prediction History")

    if "history" in st.session_state:
        st.write(pd.DataFrame(st.session_state["history"]))
    else:
        st.warning("No history yet")

    # Timestamp
    st.subheader("Timestamp")
    st.write(datetime.datetime.now())