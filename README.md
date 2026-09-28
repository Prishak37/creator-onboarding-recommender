# Creator Onboarding Recommender

A Decision Tree model + interactive app that recommends which Instagram
creators a brand should onboard for a reel/collab campaign, given a budget
and filters (gender, region, creator type, niche).

## The problem, and why it needed a re-frame

The raw dataset has no "this creator is a good pick" column - that's not
something a spreadsheet records. To turn this into a supervised learning
problem, this project engineers a **Value Tier** label:

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

## Publishing it (step-by-step)

### 1. Put the code on GitHub
1. Create a free account at github.com if you don't have one.
2. Create a new repository (e.g. `creator-onboarding-recommender`) - public,
   so employers can see it.
3. Upload these files to it: `train_model.py`, `app.py`, `requirements.txt`,
   `README.md`, `creators_scored.csv`, and the source Excel file (or just
   `creators_scored.csv` if you'd rather not publish the raw sheet).
   Easiest way if you're not comfortable with git commands: on the repo
   page, click **Add file → Upload files** and drag them in.

### 2. Deploy the app for free on Streamlit Community Cloud
1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **New app**, pick your repository, branch (`main`), and set the
   main file path to `app.py`.
3. Click **Deploy**. It builds automatically from `requirements.txt` and
   gives you a public URL like `yourname-creator-recommender.streamlit.app`.
4. Any time you push a change to GitHub, the live app updates automatically.

That URL is what goes on your CV/portfolio - a live, working ML app someone
can actually click into, not just code.

### 3. (Optional) Add it to your CV
A good one-line format:
> **Creator Onboarding Recommender** - Built a Decision Tree model
> (scikit-learn) to classify 1,900+ influencers by cost-efficiency, and
> deployed an interactive Streamlit app for budget-constrained creator
> discovery. [Live demo] · [GitHub]

## Known limitations (worth mentioning if asked in an interview)
- The label is derived from the same features used to predict it (Cost,
  Followers, Engagement Rate), so the strong feature importances on those
  three are expected, not a sign of an especially clever model - the real
  value is turning an unlabeled dataset into a usable decision-support tool.
- 73 creators (missing or zero Engagement Rate) can't be scored and are
  excluded from recommendations - shown to the user as "N/A" rather than
  silently dropped in a production version of this.
