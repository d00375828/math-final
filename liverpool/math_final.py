import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------


# Update these paths to match your machine
matches = pd.read_csv("../chinese_repo/Wimbledon_featured_matches.csv")
data_dict = pd.read_csv("../chinese_repo/data_dictionary.csv")

# --------------------------------------------------------------------
# BASIC STRUCTURE + DICTIONARY
# --------------------------------------------------------------------

print("Number of rows and columns in matches dataset:")
print(matches.shape)  # (7284, 46)

print("\nFirst 5 rows of the matches data:")
print(matches.head())

print("\nColumn names:")
print(matches.columns)

print("\nData types:")
print(matches.dtypes)

print("\nData dictionary (variable, explanation, example):")
print(data_dict)

# --------------------------------------------------------------------
# NUMERICAL SUMMARY
# --------------------------------------------------------------------

# Only numeric columns
numeric_summary = matches.describe()
print("\nNumerical summary (describe):")
print(numeric_summary)

# If you want all columns (numeric + categorical)
print("\nSummary including categorical columns:")
print(matches.describe(include="all"))

# A few key variables printed cleanly
print("\nSummary of key variables:")
for col in ["rally_count", "p1_distance_run", "p2_distance_run", "speed_mph"]:
    if col in matches.columns:
        print(f"\nSummary for {col}:")
        print(matches[col].describe())

# --------------------------------------------------------------------
# SERVING ADVANTAGE CHECK
# --------------------------------------------------------------------

# Create a simple server-won indicator (1 if server == point winner)
matches["server_won"] = (matches["server"] == matches["point_victor"]).astype(int)
server_win_rate = matches["server_won"].mean()
print(f"\nProportion of points won by the server: {server_win_rate:.3f}")

# --------------------------------------------------------------------
# GRAPHICAL SUMMARIES
# --------------------------------------------------------------------

# 1. Histogram of rally lengths
if "rally_count" in matches.columns:
    plt.figure()
    sns.histplot(matches["rally_count"], bins=30, edgecolor="black")
    plt.title("Distribution of Rally Lengths (rally_count)")
    plt.xlabel("Number of Shots in Rally")
    plt.ylabel("Number of Points")
    plt.tight_layout()
    plt.show()

# 2. Histogram of serve speeds
if "speed_mph" in matches.columns:
    plt.figure()
    # drop NaNs for plotting
    sns.histplot(matches["speed_mph"].dropna(), bins=30, edgecolor="black")
    plt.title("Distribution of Serve Speeds (mph)")
    plt.xlabel("Serve Speed (mph)")
    plt.ylabel("Number of Serves")
    plt.tight_layout()
    plt.show()

# 3. Bar chart: who wins points, server vs returner
plt.figure()
server_wins = matches["server_won"].value_counts().sort_index()
server_wins.index = ["Returner wins", "Server wins"]  # 0, 1
server_wins.plot(kind="bar", edgecolor="black")
plt.title("Point Outcomes: Server vs Returner")
plt.ylabel("Number of Points")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Bar charts for some categorical variables
categorical_cols = ["serve_width", "serve_depth", "return_depth"]

for col in categorical_cols:
    if col in matches.columns:
        plt.figure()
        matches[col].dropna().value_counts().plot(kind="bar", edgecolor="black")
        plt.title(f"Bar Chart of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# --------------------------------------------------------------------
# HEATMAP
# --------------------------------------------------------------------

# Choose a small subset of numeric variables to avoid a huge heatmap
corr_cols = [
    "rally_count",
    "p1_distance_run",
    "p2_distance_run",
    "speed_mph",
    "server_won",
]

existing_corr_cols = [c for c in corr_cols if c in matches.columns]
corr_matrix = matches[existing_corr_cols].corr()

print("\nCorrelation matrix for selected numeric variables:")
print(corr_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Selected Variables)")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------
# PREP: Sort by match and point order to compute previous results
# --------------------------------------------------------------

matches = matches.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

# We need a numeric winner column for convenience: 1 or 2 (players)
victor = matches["point_victor"]

# --------------------------------------------------------------
# 1️⃣ ONE-POINT LAG MOMENTUM TEST
# --------------------------------------------------------------

# For each match separately:
matches["prev_point_victor"] = matches.groupby("match_id")["point_victor"].shift(1)

# Now compute: Did the same player win back-to-back?
# Create outcome for the next point (for player 1)
matches["p1_win"] = (matches["point_victor"] == 1).astype(int)
matches["p1_prev_win"] = (matches["prev_point_victor"] == 1).astype(int)

# Drop the first point of each match (no previous point)
clean = matches.dropna(subset=["prev_point_victor"]).copy()

# Proportion: P(win next | won previous)
p_win_after_win = clean.loc[clean["p1_prev_win"] == 1, "p1_win"].mean()

# Proportion: P(win next | lost previous)
p_win_after_loss = clean.loc[clean["p1_prev_win"] == 0, "p1_win"].mean()

print("\n--- ONE-POINT MOMENTUM (Player 1 only for example) ---")
print(f"P(win next | won previous)  = {p_win_after_win:.3f}")
print(f"P(win next | lost previous) = {p_win_after_loss:.3f}")

# You can do the same for player 2 if needed:
matches["p2_win"] = (matches["point_victor"] == 2).astype(int)
matches["p2_prev_win"] = (matches["prev_point_victor"] == 2).astype(int)

# Make a fresh clean DataFrame *after* creating p2_prev_win
clean_p2 = matches.dropna(subset=["prev_point_victor"]).copy()

p2_win_after_win = clean_p2.loc[clean_p2["p2_prev_win"] == 1, "p2_win"].mean()
p2_win_after_loss = clean_p2.loc[clean_p2["p2_prev_win"] == 0, "p2_win"].mean()

print("\n--- Player 2 ---")
print(f"P(win next | won previous)  = {p2_win_after_win:.3f}")
print(f"P(win next | lost previous) = {p2_win_after_loss:.3f}")


# --------------------------------------------------------------
# 2️⃣ LOGISTIC REGRESSION MOMENTUM MODEL
# --------------------------------------------------------------

# Predict next-point win based on:
# - previous point outcome
# - serve indicator
# - rally length

# Create binary predictor: did player 1 win previous point?
clean["lag1"] = clean["p1_prev_win"]

# Serve indicator: 1 if player 1 served
# IMPORTANT: Adjust this column name if your dataset differs
serve_col = "server"        # check matches.columns to verify
clean["p1_serve"] = (clean[serve_col] == 1).astype(int)

# Rally length
clean["rally"] = clean["rally_count"]

# Outcome: did p1 win the point?
y = clean["p1_win"]

# Predictors:
X = clean[["lag1", "p1_serve", "rally"]]

# Fit logistic regression
log_reg = LogisticRegression()
log_reg.fit(X, y)

print("\n--- LOGISTIC REGRESSION RESULTS ---")
for name, coef in zip(["lag1", "p1_serve", "rally"], log_reg.coef_[0]):
    print(f"Coefficient for {name}: {coef:.4f}")

print(f"Intercept: {log_reg.intercept_[0]:.4f}")

# Interpretation notes:
# - Positive coefficient for lag1 means: winning previous point increases
#   the probability of winning next point (momentum).
# - Positive coefficient for p1_serve means serving increases win likelihood.
# - rally coefficient can help identify if longer rallies favor a player.

# --------------------------------------------------------------
# 3️⃣ STREAK-BASED ANALYSIS (MULTI-POINT MOMENTUM)
# --------------------------------------------------------------

# Define a streak: consecutive points won by the same player (player 1 example)

def compute_streaks(x):
    streaks = []
    current = 0
    for val in (x == 1):  # True if p1 wins, False otherwise
        if val:
            current += 1
        else:
            current = 0
        streaks.append(current)
    return streaks

matches["p1_streak"] = matches.groupby("match_id")["p1_win"].transform(compute_streaks)

# To compute P(win next | current streak length k)
clean2 = matches[matches["p1_streak"].notna()].copy()
clean2["next_p1_win"] = clean2.groupby("match_id")["p1_win"].shift(-1)

prob_by_streak = clean2.groupby("p1_streak")["next_p1_win"].mean()

print("\n--- STREAK-BASED MOMENTUM ---")
print(prob_by_streak)

# Plot it
plt.figure(figsize=(8,5))
prob_by_streak.plot(marker="o")
plt.title("Probability of Winning Next Point vs. Current Streak Length (Player 1)")
plt.xlabel("Current Streak Length")
plt.ylabel("P(win next point)")
plt.grid(True)
plt.show()







# -----------------------------------------------------
# Load the Wimbledon data
# -----------------------------------------------------
# wim = pd.read_csv("/Users/thegoat/Desktop/Math-2050/2024_MCM_Problem_C_Data/2024_Wimbledon_featured_matches.csv")
wim = matches
# -----------------------------------------------------
# 1. Keep only numeric columns
# -----------------------------------------------------
numeric_cols = wim.select_dtypes(include=[np.number]).columns.tolist()

# Choose the target variable (what you want to predict)
# You can change this to something else like 'rally_count' if needed
y_var = 'point_victor'

# Make sure the target is in the numeric columns
if y_var not in numeric_cols:
    raise ValueError(f"{y_var} is not numeric or not found in the dataframe.")

# Predictor columns = all numeric columns except the target
X_cols = [col for col in numeric_cols if col != y_var]

# -----------------------------------------------------
# 2. Build clean dataset with y and X, dropping rows with any missing values
# -----------------------------------------------------
data = wim[[y_var] + X_cols].dropna()

y = data[y_var]
X = data[X_cols]

# -----------------------------------------------------
# 3. Add constant and fit ONE OLS model
# -----------------------------------------------------
X_const = sm.add_constant(X)

full_model = sm.OLS(y, X_const).fit()

print(full_model.summary())




# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
# import statsmodels.api as sm

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
matches = pd.read_csv("../chinese_repo/Wimbledon_featured_matches.csv")

# ------------------------------------------------------
# Define the final match
# ------------------------------------------------------
final_id = "2023-wimbledon-1701"

train_df = matches[matches["match_id"] != final_id].copy()
test_df  = matches[matches["match_id"] == final_id].copy()

# ------------------------------------------------------
# Define binary target for logistic regression
# ------------------------------------------------------
train_df["y"] = (train_df["point_victor"] == 1).astype(int)
test_df["y"]  = (test_df["point_victor"] == 1).astype(int)

# ------------------------------------------------------
# Choose predictor variables (no leakage!)
# ------------------------------------------------------
predictors = [
    "server",              # 1 or 2 → convert to p1_serving later
    "serve_no",
    "rally_count",
    "speed_mph",
    "p1_ace","p2_ace",
    "p1_winner","p2_winner",
    "p1_unf_err","p2_unf_err",
    "p1_double_fault","p2_double_fault",
    "p1_distance_run","p2_distance_run",
    "p1_break_pt","p2_break_pt"
]

# Keep only predictors that exist in the dataset
predictors = [col for col in predictors if col in matches.columns]

# ------------------------------------------------------
# Prepare training data
# ------------------------------------------------------
train = train_df[["y"] + predictors].dropna().copy()
test  = test_df[["y"] + predictors].dropna().copy()

# Convert server → p1_serve indicator
train["p1_serve"] = (train["server"] == 1).astype(int)
test["p1_serve"]  = (test["server"] == 1).astype(int)

train = train.drop(columns=["server"])
test  = test.drop(columns=["server"])

# X and y
X_train = train.drop(columns=["y"])
y_train = train["y"]

X_test = test.drop(columns=["y"])
y_test = test["y"]

# ------------------------------------------------------
# Step 1: Fit initial logistic model using statsmodels
# ------------------------------------------------------
X_train_sm = sm.add_constant(X_train)
initial_model = sm.Logit(y_train, X_train_sm).fit()
print(initial_model.summary())

# ------------------------------------------------------
# Step 2: Remove features with p > 0.05
# ------------------------------------------------------
significant_predictors = initial_model.pvalues[initial_model.pvalues < 0.05].index.tolist()

# Remove constant if included
significant_predictors = [p for p in significant_predictors if p != "const"]

print("\nSignificant predictors:", significant_predictors)

# ------------------------------------------------------
# Step 3: Fit optimized logistic regression (sklearn)
# ------------------------------------------------------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train[significant_predictors], y_train)

# ------------------------------------------------------
# Step 4: Test on the withheld final match
# ------------------------------------------------------
y_pred = log_reg.predict(X_test[significant_predictors])

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- PERFORMANCE ON FINAL MATCH ---")
print(f"Accuracy on final: {acc:.3f}")
print("\nConfusion Matrix:")
print(cm)

# Optional: print coefficients
coef_table = pd.DataFrame({
    "predictor": significant_predictors,
    "coef": log_reg.coef_[0]
})
print("\nOptimized model coefficients:")
print(coef_table)
