"""
train_model.py
--------------
Cleans Influencer_recommender_csv.csv, engineers a "Value Tier" label, trains a
Decision Tree, and writes the two files the Streamlit app needs:

    creators_scored.csv  - clean data + predicted value tier + probabilities
    model.pkl            - the trained model (for reference / reuse)

Run:  python train_model.py

WHY A "VALUE TIER" LABEL
The data has no column saying "this creator is a good pick", so we engineer one.
Value = cost efficiency = Reel Cost / (Followers x Engagement Rate), i.e. rupees
paid per engaged follower (lower is better). Creators are ranked only against
peers in the SAME Creator Type, because price scales with size and comparing a
Nano to a Mega/Celebrity would be meaningless. Each tier is split into thirds:
High / Medium / Low Value.
"""

import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

SOURCE_FILE = "Influencer_recommender_csv.csv"

NUMERIC_COLS = [
    "Followers",
    "Engagement Rate (%)",
    "Average Views",
    "Average Reach",
    "Story Cost",
    "Reel / Collab Reel Cost",
    "Carousel Post Cost",
    "Digital / Ad Rights Cost",
]
CATEGORICAL_COLS = ["Creator Type", "Gender", "Region", "Niche Category (Standardized)"]
FEATURE_COLS = [
    "Followers",
    "Engagement Rate (%)",
    "Average Views",
    "Average Reach",
    "Reel / Collab Reel Cost",
] + CATEGORICAL_COLS

TARGET = "Value Tier"
TIER_LABELS = ["High Value", "Medium Value", "Low Value"]
MAX_PLAUSIBLE_ER = 50  # engagement rates above this (up to 100%) are treated as data errors


def to_number(series: pd.Series) -> pd.Series:
    """'4,97,00,000' -> 49700000 ; '1.89%' -> 1.89 ; ' 8,00,000 ' -> 800000."""
    cleaned = series.astype(str).str.replace(r"[,%\s₹]", "", regex=True)
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_clean(path: str = SOURCE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]   # ' Reel / Collab Reel Cost' -> 'Reel / Collab Reel Cost'
    df = df.dropna(how="all")                       # the CSV has ~2,300 empty trailing rows
    df = df.dropna(subset=["Name", "Reel / Collab Reel Cost"])

    for col in NUMERIC_COLS:
        df[col] = to_number(df[col])

    for col in CATEGORICAL_COLS + ["Final_cities"]:
        df[col] = df[col].astype(str).str.strip()

    return df.reset_index(drop=True)


def add_value_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Cost per Engaged Follower and the Value Tier label (NaN where it can't be computed)."""
    df = df.copy()
    scorable = (
        df["Followers"].gt(0)
        & df["Engagement Rate (%)"].gt(0)
        & df["Engagement Rate (%)"].le(MAX_PLAUSIBLE_ER)
        & df["Reel / Collab Reel Cost"].gt(0)
    )
    df["Cost per Engaged Follower"] = np.nan
    df.loc[scorable, "Cost per Engaged Follower"] = df.loc[scorable, "Reel / Collab Reel Cost"] / (
        df.loc[scorable, "Followers"] * df.loc[scorable, "Engagement Rate (%)"] / 100
    )

    def tier(group: pd.Series) -> pd.Series:
        if group.notna().sum() < 3:
            return pd.Series(np.nan, index=group.index, dtype="object")
        ranks = group.rank(method="first", pct=True)   # rank-based so duplicate values never break the split
        out = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0], labels=TIER_LABELS, include_lowest=True)
        return out.astype("object")

    df[TARGET] = df.groupby("Creator Type")["Cost per Engaged Follower"].transform(tier)
    return df


def encode(df: pd.DataFrame, encoders: dict | None = None):
    X = df[FEATURE_COLS].copy()
    fitted = encoders is not None
    encoders = encoders or {}
    for col in CATEGORICAL_COLS:
        if not fitted:
            encoders[col] = LabelEncoder().fit(X[col].astype(str))
        X[col] = encoders[col].transform(X[col].astype(str))
    return X, encoders


def main():
    df = add_value_label(load_and_clean())
    labelled = df[df[TARGET].notna()].copy()
    print(f"Clean creators: {len(df)} | usable for training: {len(labelled)} | not scorable: {len(df) - len(labelled)}")
    print(labelled[TARGET].value_counts().to_string(), "\n")

    X, encoders = encode(labelled)
    y = labelled[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Hold-out accuracy: {accuracy_score(y_test, pred):.3f}   (random guessing = 0.333)")
    cv = cross_val_score(
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42), X, y, cv=5
    )
    print(f"5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}\n")
    print(classification_report(y_test, pred))

    print("Feature importances:")
    for col, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {col:32s} {imp:.3f}")

    # Refit on ALL labelled data for the shipped model, then score every scorable creator
    model.fit(X, y)
    scorable = df[TARGET].notna()
    X_all, _ = encode(df[scorable], encoders)
    df["Predicted Value Tier"] = "Not scored"
    df.loc[scorable, "Predicted Value Tier"] = model.predict(X_all)

    proba = pd.DataFrame(model.predict_proba(X_all), columns=model.classes_, index=X_all.index)
    df["P(High Value)"] = np.nan
    df.loc[scorable, "P(High Value)"] = proba["High Value"].round(3)

    df.to_csv("creators_scored.csv", index=False)
    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "feature_cols": FEATURE_COLS}, f)
    print("\nSaved creators_scored.csv and model.pkl")


if __name__ == "__main__":
    main()
