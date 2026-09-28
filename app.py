"""
app.py
------
Streamlit app: a brand/marketer picks their filters (budget, gender, city/region,
creator type) and the app returns the top recommended creators, ranked by a
Decision Tree model's predicted "Value Tier" (High / Medium / Low cost-efficiency).

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Creator Onboarding Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(r"Influencer recommender csv.csv")
    return df

df = load_data()

st.title("🎯 Creator Onboarding Recommender")
st.caption(
    "A Decision Tree model scores each Instagram creator's cost-efficiency "
    "(engagement delivered per rupee spent) against peers in the same tier. "
    "Set your campaign filters below to get ranked recommendations."
)

# ---------------- Sidebar filters ----------------
st.sidebar.header("Campaign Filters")

max_budget = st.sidebar.number_input(
    "Max Reel/Collab budget (₹)",
    min_value=0,
    value=100000,
    step=5000,
)

creator_types = st.sidebar.multiselect(
    "Creator Type",
    options=sorted(df["Creator Type"].dropna().unique()),
    default=sorted(df["Creator Type"].dropna().unique()),
)

genders = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["Gender"].dropna().unique()),
    default=sorted(df["Gender"].dropna().unique()),
)

regions = st.sidebar.multiselect(
    "Region",
    options=sorted(df["Region"].dropna().unique()),
    default=sorted(df["Region"].dropna().unique()),
)

niches = st.sidebar.multiselect(
    "Niche",
    options=sorted(df["Niche Category (Standardized)"].dropna().unique()),
    default=sorted(df["Niche Category (Standardized)"].dropna().unique()),
)

value_tiers = st.sidebar.multiselect(
    "Predicted Value Tier",
    options=["High Value", "Medium Value", "Low Value"],
    default=["High Value", "Medium Value"],
)

top_n = st.sidebar.slider("How many creators to show", 5, 50, 15)

# ---------------- Filtering ----------------
filtered = df[
    (df["Reel / Collab Reel Cost"] <= max_budget)
    & (df["Creator Type"].isin(creator_types))
    & (df["Gender"].isin(genders))
    & (df["Region"].isin(regions))
    & (df["Niche Category (Standardized)"].isin(niches))
    & (df["Predicted Value Tier"].isin(value_tiers))
].copy()

tier_order = {"High Value": 0, "Medium Value": 1, "Low Value": 2}
filtered["_sort"] = filtered["Predicted Value Tier"].map(tier_order)
filtered = filtered.sort_values(["_sort", "Followers"], ascending=[True, False]).drop(columns="_sort")

# ---------------- Results ----------------
st.subheader(f"{len(filtered)} creators match your filters")

if len(filtered) == 0:
    st.warning("No creators match these filters — try relaxing the budget or filters.")
else:
    display_cols = [
        "Name", "Creator Type", "Gender", "Final_cities", "Region",
        "Niche Category (Standardized)", "Followers", "Engagement Rate (%)",
        "Reel / Collab Reel Cost", "Predicted Value Tier",
    ]
    st.dataframe(
        filtered[display_cols].head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download full filtered list (CSV)",
        filtered[display_cols].to_csv(index=False),
        file_name="recommended_creators.csv",
    )

with st.expander("How the model works"):
    st.markdown(
        """
        1. **Label engineering**: each creator's *Cost per Engaged Follower*
           (`Reel Cost / (Followers × Engagement Rate)`) is compared only against
           peers in the same Creator Type tier, then split into three equal
           groups: High / Medium / Low Value.
        2. **Model**: a `DecisionTreeClassifier` (scikit-learn, max depth 5) is
           trained on Followers, Engagement Rate, Average Views, Average Reach,
           Creator Type, Gender, Region, Niche, and Reel Cost to predict that tier.
        3. **Recommendation**: filtering the dataset to your campaign
           constraints, then ranking by predicted tier (High → Low) and
           follower count.
        """
    )
