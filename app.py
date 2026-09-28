"""
app.py - Creator Onboarding Recommender (Streamlit)

A brand sets its campaign constraints (budget, gender, region, city, creator type,
niche) and the app returns creators ranked by a Decision Tree's predicted
cost-efficiency ("Value Tier") for a reel/collab.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Creator Onboarding Recommender", page_icon="🎯", layout="wide")

DATA_FILE = Path(__file__).parent / "creators_scored.csv"
TIER_RANK = {"High Value": 0, "Medium Value": 1, "Low Value": 2, "Not scored": 3}
COST = "Reel / Collab Reel Cost"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


if not DATA_FILE.exists():
    st.error("creators_scored.csv not found. Run `python train_model.py` first.")
    st.stop()

df = load_data()

st.title("🎯 Creator Onboarding Recommender")
st.caption(
    "A Decision Tree scores each Instagram creator's cost-efficiency (engagement delivered per "
    "rupee spent) against peers of the same size tier. Set your campaign filters on the left."
)

# ---------------- Sidebar filters ----------------
st.sidebar.header("Campaign filters")

max_cost = int(df[COST].max())
budget = st.sidebar.number_input(
    "Max budget per reel/collab (₹)", min_value=0, max_value=max_cost, value=100_000, step=5_000
)


def multi(label: str, column: str, default_all: bool = True):
    options = sorted(df[column].dropna().unique())
    return st.sidebar.multiselect(label, options, default=options if default_all else None)


creator_types = multi("Creator type", "Creator Type")
genders = multi("Gender", "Gender")
regions = multi("Region", "Region")

city_options = sorted(df[df["Region"].isin(regions)]["Final_cities"].dropna().unique())
cities = st.sidebar.multiselect("City (leave empty for all)", city_options)

niches = multi("Niche", "Niche Category (Standardized)")

tier_options = ["High Value", "Medium Value", "Low Value", "Not scored"]
tiers = st.sidebar.multiselect(
    "Predicted value tier",
    tier_options,
    default=["High Value", "Medium Value"],
    help="'Not scored' = creators with missing/implausible engagement rate, so no prediction.",
)
top_n = st.sidebar.slider("Creators to show", 5, 100, 20)

# ---------------- Filtering + ranking ----------------
mask = (
    (df[COST] <= budget)
    & df["Creator Type"].isin(creator_types)
    & df["Gender"].isin(genders)
    & df["Region"].isin(regions)
    & df["Niche Category (Standardized)"].isin(niches)
    & df["Predicted Value Tier"].isin(tiers)
)
if cities:
    mask &= df["Final_cities"].isin(cities)

results = df[mask].copy()
results["_tier"] = results["Predicted Value Tier"].map(TIER_RANK)
results = results.sort_values(
    ["_tier", "P(High Value)", "Followers"], ascending=[True, False, False]
).drop(columns="_tier")

# ---------------- Output ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Creators matching", f"{len(results):,}")
c2.metric("Avg reel cost", f"₹{results[COST].mean():,.0f}" if len(results) else "–")
c3.metric("Avg engagement rate", f"{results['Engagement Rate (%)'].mean():.2f}%" if len(results) else "–")

if results.empty:
    st.warning("No creators match these filters. Try raising the budget or relaxing a filter.")
else:
    show = results[
        [
            "Name", "Instagram Link", "Creator Type", "Gender", "Final_cities", "Region",
            "Niche Category (Standardized)", "Followers", "Engagement Rate (%)",
            COST, "Predicted Value Tier", "P(High Value)",
        ]
    ].rename(columns={"Final_cities": "City", "Niche Category (Standardized)": "Niche"})

    st.dataframe(
        show.head(top_n),
        hide_index=True,
        column_config={
            "Instagram Link": st.column_config.LinkColumn("Instagram", display_text="open"),
            "Followers": st.column_config.NumberColumn(format="%d"),
            COST: st.column_config.NumberColumn("Reel cost (₹)", format="%d"),
            "P(High Value)": st.column_config.ProgressColumn(
                "P(High Value)", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    st.download_button(
        "Download all matches (CSV)",
        show.to_csv(index=False),
        file_name="recommended_creators.csv",
        mime="text/csv",
    )

with st.expander("How this works"):
    st.markdown(
        """
**1. Label engineering.** The data has no "good pick" column, so one is built:
*Cost per Engaged Follower* = `Reel Cost ÷ (Followers × Engagement Rate)`. Each creator is compared
only with peers in the same Creator Type, then split into thirds: **High / Medium / Low Value**.

**2. Model.** A scikit-learn `DecisionTreeClassifier` (max depth 5) learns to predict that tier from
followers, engagement rate, views, reach, cost, creator type, gender, region and niche.

**3. Recommendation.** Your filters narrow the list; creators are ranked by predicted tier, then by the
model's probability of being *High Value*, then by followers.

**Limits.** The label is computed from cost, followers and engagement, so those features dominate the
tree by construction. Creators with missing or implausible (>50%) engagement rate can't be scored.
        """
    )
