import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="NFL Player Value Engine", page_icon="🏈", layout="wide")


# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("data/model_df.csv")


@st.cache_resource
def load_model():
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le


df = load_data()
model, le = load_model()

# Title
st.title("🏈 NFL Player Value Engine")
st.markdown(
    "*Predicting what NFL players should earn based on performance — built by Darren Barkins*"
)
st.divider()

# Sidebar filters
st.sidebar.header("🔍 Filter Players")
positions = ["All"] + sorted(df["position"].unique().tolist())
selected_position = st.sidebar.selectbox("Select Position", positions)
seasons = ["All"] + sorted(df["season"].unique().tolist())
selected_season = st.sidebar.selectbox("Select Season", seasons)

# Apply filters
filtered_df = df.copy()
if selected_position != "All":
    filtered_df = filtered_df[filtered_df["position"] == selected_position]
if selected_season != "All":
    filtered_df = filtered_df[filtered_df["season"] == selected_season]

# ── Section 1: Key Metrics ──
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Players", filtered_df["player_name"].nunique())
col2.metric("Avg Actual AAV", f"${filtered_df['aav'].mean():.1f}M")
col3.metric("Avg Predicted AAV", f"${filtered_df['predicted_aav'].mean():.1f}M")
col4.metric(
    "Seasons", f"{int(filtered_df['season'].min())}–{int(filtered_df['season'].max())}"
)

st.divider()

# ── Section 2: Overpaid vs Underpaid ──
st.subheader("💰 Most Overpaid vs Underpaid Players")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 Most Overpaid")
    overpaid = (
        filtered_df.groupby("player_name")["value_delta"]
        .mean()
        .reset_index()
        .nlargest(10, "value_delta")
    )
    fig = px.bar(
        overpaid,
        x="value_delta",
        y="player_name",
        orientation="h",
        color="value_delta",
        color_continuous_scale="Reds",
        labels={"value_delta": "Value Delta ($M)", "player_name": "Player"},
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("#### 🟢 Most Underpaid")
    underpaid = (
        filtered_df.groupby("player_name")["value_delta"]
        .mean()
        .reset_index()
        .nsmallest(10, "value_delta")
    )
    fig2 = px.bar(
        underpaid,
        x="value_delta",
        y="player_name",
        orientation="h",
        color="value_delta",
        color_continuous_scale="Greens_r",
        labels={"value_delta": "Value Delta ($M)", "player_name": "Player"},
    )
    fig2.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total descending"},
    )
    st.plotly_chart(fig2, width="stretch")

st.divider()

# ── Section 3: Player Search ──
st.subheader("🔎 Search a Player")
player_search = st.selectbox(
    "Select a player", sorted(df["player_name"].unique().tolist())
)
player_data = df[df["player_name"] == player_search]

if not player_data.empty:
    latest = player_data.sort_values("season").iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Position", latest["position"])
    col2.metric("Actual AAV", f"${latest['aav']:.1f}M")
    col3.metric(
        "Predicted AAV",
        f"${latest['predicted_aav']:.1f}M",
        delta=f"${latest['value_delta']:.1f}M",
        delta_color="inverse",
    )

    st.dataframe(
        player_data[
            ["season", "aav", "predicted_aav", "value_delta", "fantasy_points", "age"]
        ].sort_values("season"),
        width="stretch",
    )

st.divider()

# ── Section 4: What-If Simulator ──
st.subheader("🔮 What-If Salary Simulator")
st.markdown("*Adjust a player's stats and see how their predicted salary changes*")

col1, col2 = st.columns(2)

with col1:
    sim_position = st.selectbox("Position", sorted(df["position"].unique().tolist()))
    sim_age = st.slider("Age", 21, 40, 26)
    sim_years_exp = st.slider("Years of Experience", 0, 15, 3)
    sim_games = st.slider("Games Played", 1, 17, 16)
    sim_draft = st.slider("Draft Pick #", 1, 300, 50)

with col2:
    sim_fantasy = st.slider("Fantasy Points", 0, 400, 150)
    sim_pass_yards = st.slider("Passing Yards", 0, 5000, 0)
    sim_pass_tds = st.slider("Passing TDs", 0, 50, 0)
    sim_rush_yards = st.slider("Rushing Yards", 0, 2000, 0)
    sim_rec_yards = st.slider("Receiving Yards", 0, 2000, 0)

# Encode position
sim_position_encoded = le.transform([sim_position])[0]

# Build input for model
sim_input = pd.DataFrame(
    [
        {
            "fantasy_points": sim_fantasy,
            "rushing_yards": sim_rush_yards,
            "rushing_tds": 0,
            "receiving_yards": sim_rec_yards,
            "receiving_tds": 0,
            "receptions": 0,
            "targets": 0,
            "passing_yards": sim_pass_yards,
            "passing_tds": sim_pass_tds,
            "interceptions": 0,
            "age": sim_age,
            "years_exp": sim_years_exp,
            "draft_overall": sim_draft,
            "games": sim_games,
            "position_encoded": sim_position_encoded,
        }
    ]
)

# Predict
sim_prediction = model.predict(sim_input)[0]

st.markdown("---")
st.metric(
    label="💰 Predicted Fair Market Value", value=f"${sim_prediction:.1f}M per year"
)
st.caption(
    "Adjust the sliders above to see how performance changes predicted salary in real time"
)
