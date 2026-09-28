# Creator Onboarding Recommender

A Decision Tree model + interactive app that recommends which Instagram
creators a brand should onboard for a reel/collab campaign, given a budget
and filters (gender, region, creator type, niche).

**Cost per Engaged Follower** = `Reel/Collab Cost ÷ (Followers × Engagement Rate)`

This is compared **within each Creator Type tier only** (Nano vs Nano,
Macro vs Macro, etc.) because price scales naturally with follower count -
comparing a Nano creator's price to a Mega/Celebrity's would be meaningless.
Each tier is split into three equal groups: **High / Medium / Low Value**.

## Model

A `DecisionTreeClassifier` (scikit-learn) predicts that Value Tier from:
Followers, Engagement Rate, Average Views, Average Reach, Creator Type,
Gender, Region, Niche, and Reel Cost.

- Test accuracy: **~63%** on a 3-class problem (random baseline: 33%)
- Max depth capped at 5 so the tree stays interpretable enough to walk
  through in an interview
- Feature importance confirms Engagement Rate, Reel Cost, and Followers
  drive the prediction (expected, since they define the label) - a fair
  callout for anyone reviewing this project

## How it's used

The Streamlit app lets a user set: max budget, gender, region, creator
type, and niche. It filters the dataset to matches, then ranks by the
model's predicted Value Tier (High first) and follower count, and lets
the user download the shortlist as a CSV.

## Running locally

```bash
pip install -r requirements.txt
python train_model.py      # trains the model, writes model.pkl + creators_scored.csv
streamlit run app.py       # launches the app at http://localhost:8501
```
