"""
train_model.py
---------------
Trains a Decision Tree classifier that labels each Instagram creator as
High / Medium / Low VALUE for a brand deciding who to onboard for a reel/collab.

WHY A "VALUE TIER" LABEL:
The raw data has no column that says "this creator is a good pick" - that
label has to be engineered. We define value as cost efficiency: how much a
brand pays per engaged follower reached (Reel Cost / (Followers x Engagement
Rate)). Lower cost-per-engaged-follower = better value. We rank creators
against peers in the SAME Creator Type tier (Nano/Micro/Macro/Mega) because
price naturally scales with follower count - comparing a Nano creator's
price to a Mega/Celebrity's would be meaningless.

Run:
    python train_model.py
Produces:
    model.pkl          - the trained DecisionTreeClassifier + encoders, bundled
    creators_scored.csv - full dataset with predicted value tier, for the app
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

SOURCE_FILE = "Combined_Creator_Database_ReelCity_Sorted_Filtered__1_.xlsx"

FEATURE_COLS = [
    "Followers",
    "Engagement Rate (%)",
    "Average Views",
    "Average Reach",
    "Creator Type",
    "Gender",
    "Region",
    "Niche Category (Standardized)",
    "Reel / Collab Reel Cost",
]
CATEGORICAL_COLS = ["Creator Type", "Gender", "Region", "Niche Category (Standardized)"]
TARGET_COL = "Value Tier"


def load_and_engineer():
    df = pd.read_excel(SOURCE_FILE, sheet_name="Instagram")

    # Can't compute a value score without a real engagement rate
    df = df[df["Engagement Rate (%)"].notna() & (df["Engagement Rate (%)"] > 0)].copy()

    df["CostPerEngagedFollower"] = df["Reel / Collab Reel Cost"] / (
        df["Followers"] * (df["Engagement Rate (%)"] / 100)
    )

    # Rank into 3 tiers WITHIN each Creator Type (lower cost-per-engaged-follower = better)
    def tier_within_group(group):
        # qcut can fail on small groups with duplicate edges; fall back to rank-based split
        try:
            return pd.qcut(group, 3, labels=["High Value", "Medium Value", "Low Value"])
        except ValueError:
            ranks = group.rank(method="first")
            n = len(group)
            bins = pd.cut(ranks, 3, labels=["High Value", "Medium Value", "Low Value"])
            return bins

    df[TARGET_COL] = df.groupby("Creator Type")["CostPerEngagedFollower"].transform(tier_within_group)
    df = df.dropna(subset=[TARGET_COL])

    return df


def train():
    df = load_and_engineer()
    print(f"Training rows after cleaning: {len(df)}")
    print(df[TARGET_COL].value_counts())

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(str)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # max_depth=5 keeps the tree small enough to actually explain in an interview
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    print("\nFeature importances:")
    for col, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {col}: {imp:.3f}")

    # Score the FULL dataset (including the 62 rows we couldn't label) for the app to use
    full_df = pd.read_excel(SOURCE_FILE, sheet_name="Instagram")
    X_full = full_df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        # unseen categories -> -1 so predict() doesn't crash; the app will filter these out
        known = set(encoders[col].classes_)
        X_full[col] = X_full[col].astype(str).apply(lambda v: v if v in known else encoders[col].classes_[0])
        X_full[col] = encoders[col].transform(X_full[col])

    valid_rows = full_df["Engagement Rate (%)"].notna() & (full_df["Engagement Rate (%)"] > 0)
    full_df["Predicted Value Tier"] = None
    full_df.loc[valid_rows, "Predicted Value Tier"] = model.predict(X_full[valid_rows])

    full_df.to_csv("creators_scored.csv", index=False)

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "feature_cols": FEATURE_COLS}, f)

    print("\nSaved model.pkl and creators_scored.csv")


if __name__ == "__main__":
    train()
