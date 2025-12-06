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
# DATA INTRODUCTION: MOMENTUM-RELEVANT VARIABLES
# --------------------------------------------------------------------

print("="*80)
print("DATA INTRODUCTION: KEY PREDICTORS FOR MOMENTUM ANALYSIS")
print("="*80)

print("\nNumber of rows and columns in matches dataset:")
print(matches.shape)  # (7284, 46)

print("\nData dictionary (variable, explanation, example):")
print(data_dict)

# Sort data for proper analysis
matches = matches.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

# --------------------------------------------------------------------
# CREATE MOMENTUM-RELEVANT FEATURES FOR EXPLORATION
# --------------------------------------------------------------------

# Create server-won indicator
matches["server_won"] = (matches["server"] == matches["point_victor"]).astype(int)

# Create serving indicators
matches["p1_serve"] = (matches["server"] == 1).astype(int)
matches["p2_serve"] = (matches["server"] == 2).astype(int)

# Convert score to numeric for analysis
matches["p1_score_numeric"] = matches["p1_score"].map({"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}).fillna(0)
matches["p2_score_numeric"] = matches["p2_score"].map({"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}).fillna(0)
matches["score_diff"] = matches["p1_score_numeric"] - matches["p2_score_numeric"]

# Game and set context
matches["p1_games_ahead"] = matches["p1_games"] - matches["p2_games"]
matches["p1_sets_ahead"] = matches["p1_sets"] - matches["p2_sets"]

# Break point indicators
matches["is_break_point"] = ((matches["p1_break_pt"] == 1) | (matches["p2_break_pt"] == 1)).astype(int)
matches["p1_has_break_pt"] = (matches["p1_break_pt"] == 1).astype(int)
matches["p2_has_break_pt"] = (matches["p2_break_pt"] == 1).astype(int)

# Point outcomes
matches["p1_win"] = (matches["point_victor"] == 1).astype(int)
matches["p2_win"] = (matches["point_victor"] == 2).astype(int)

# --------------------------------------------------------------------
# SUMMARY STATISTICS FOR KEY PREDICTORS
# --------------------------------------------------------------------

print("\n" + "="*80)
print("SUMMARY STATISTICS: KEY PREDICTORS FOR MOMENTUM")
print("="*80)

# 1. Serving Advantage
server_win_rate = matches["server_won"].mean()
p1_serve_rate = matches["p1_serve"].mean()
print(f"\n1. SERVING ADVANTAGE:")
print(f"   Overall server win rate: {server_win_rate:.3f} ({server_win_rate*100:.1f}%)")
print(f"   Player 1 serves: {p1_serve_rate:.3f} ({p1_serve_rate*100:.1f}%) of points")
print(f"   Player 2 serves: {1-p1_serve_rate:.3f} ({(1-p1_serve_rate)*100:.1f}%) of points")

# Server win rate by player
p1_serve_win = matches[matches["p1_serve"] == 1]["p1_win"].mean()
p2_serve_win = matches[matches["p2_serve"] == 1]["p2_win"].mean()
print(f"   When P1 serves, P1 wins: {p1_serve_win:.3f}")
print(f"   When P2 serves, P2 wins: {p2_serve_win:.3f}")

# 2. Score Context
print(f"\n2. GAME SCORE CONTEXT:")
print(f"   Mean score difference (P1 - P2): {matches['score_diff'].mean():.3f}")
print(f"   Std dev of score difference: {matches['score_diff'].std():.3f}")
print(f"   Score difference range: [{matches['score_diff'].min():.0f}, {matches['score_diff'].max():.0f}]")
print(f"   Points with break point opportunity: {matches['is_break_point'].mean():.3f} ({matches['is_break_point'].mean()*100:.1f}%)")

# 3. Match Context
print(f"\n3. MATCH CONTEXT:")
print(f"   Mean games ahead (P1 - P2): {matches['p1_games_ahead'].mean():.3f}")
print(f"   Mean sets ahead (P1 - P2): {matches['p1_sets_ahead'].mean():.3f}")
print(f"   Games ahead range: [{matches['p1_games_ahead'].min():.0f}, {matches['p1_games_ahead'].max():.0f}]")
print(f"   Sets ahead range: [{matches['p1_sets_ahead'].min():.0f}, {matches['p1_sets_ahead'].max():.0f}]")

# 4. Performance Metrics
print(f"\n4. PERFORMANCE METRICS:")
if "rally_count" in matches.columns:
    print(f"   Mean rally length: {matches['rally_count'].mean():.2f} shots")
    print(f"   Median rally length: {matches['rally_count'].median():.2f} shots")
    print(f"   Rally length range: [{matches['rally_count'].min()}, {matches['rally_count'].max()}]")
if "speed_mph" in matches.columns:
    print(f"   Mean serve speed: {matches['speed_mph'].mean():.1f} mph")
    print(f"   Serve speed range: [{matches['speed_mph'].min():.0f}, {matches['speed_mph'].max():.0f}] mph")
if "p1_distance_run" in matches.columns:
    print(f"   Mean distance run (P1): {matches['p1_distance_run'].mean():.2f} meters")
    print(f"   Mean distance run (P2): {matches['p2_distance_run'].mean():.2f} meters")

# 5. Point Outcomes
print(f"\n5. POINT OUTCOMES:")
print(f"   Player 1 win rate: {matches['p1_win'].mean():.3f} ({matches['p1_win'].mean()*100:.1f}%)")
print(f"   Player 2 win rate: {matches['p2_win'].mean():.3f} ({matches['p2_win'].mean()*100:.1f}%)")

# Save summary statistics
summary_stats = pd.DataFrame({
    "Predictor": [
        "Server Win Rate", "P1 Serve Rate", "P1 Serve Win Rate", "P2 Serve Win Rate",
        "Mean Score Diff", "Break Point Rate",
        "Mean Games Ahead", "Mean Sets Ahead",
        "Mean Rally Length", "Mean Serve Speed",
        "P1 Win Rate"
    ],
    "Value": [
        server_win_rate, p1_serve_rate, p1_serve_win, p2_serve_win,
        matches['score_diff'].mean(), matches['is_break_point'].mean(),
        matches['p1_games_ahead'].mean(), matches['p1_sets_ahead'].mean(),
        matches['rally_count'].mean() if "rally_count" in matches.columns else None,
        matches['speed_mph'].mean() if "speed_mph" in matches.columns else None,
        matches['p1_win'].mean()
    ]
})
summary_stats.to_csv("data_intro_summary_stats.csv", index=False)

# --------------------------------------------------------------------
# VISUALIZATIONS: MOMENTUM-RELEVANT PREDICTORS
# --------------------------------------------------------------------

print("\n" + "="*80)
print("CREATING DATA INTRODUCTION VISUALIZATIONS")
print("="*80)

# 1.1 Server win rate
plt.figure(figsize=(8, 6))
server_wins = matches["server_won"].value_counts().sort_index()
plt.bar(["Returner Wins", "Server Wins"], 
        [server_wins.get(0, 0), server_wins.get(1, 0)], 
        color=['orange', 'green'], edgecolor='black', alpha=0.7)
plt.axhline(len(matches)/2, color='red', linestyle='--', linewidth=2, 
            label='50% (no advantage)')
plt.ylabel('Number of Points')
plt.title(f'Serving Advantage\n(Server wins {server_win_rate*100:.1f}% of points)', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("data_intro_serving_advantage.png", dpi=300, bbox_inches='tight')
plt.show()

# 1.2 Score difference distribution
plt.figure(figsize=(8, 6))
plt.hist(matches["score_diff"], bins=21, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Tied (0)')
plt.xlabel('Score Difference (P1 - P2)')
plt.ylabel('Frequency')
plt.title('Distribution of Score Differences Within Games', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("data_intro_score_difference.png", dpi=300, bbox_inches='tight')
plt.show()

# 1.3 Games ahead distribution
plt.figure(figsize=(8, 6))
plt.hist(matches["p1_games_ahead"], bins=15, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Tied (0)')
plt.xlabel('Games Ahead (P1 - P2)')
plt.ylabel('Frequency')
plt.title('Distribution of Games Ahead in Current Set', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("data_intro_games_ahead.png", dpi=300, bbox_inches='tight')
plt.show()

# 1.4 Break point frequency
plt.figure(figsize=(8, 6))
break_pt_counts = matches["is_break_point"].value_counts().sort_index()
plt.bar(["Regular Point", "Break Point"], 
        [break_pt_counts.get(0, 0), break_pt_counts.get(1, 0)], 
        color=['lightblue', 'red'], edgecolor='black', alpha=0.7)
plt.ylabel('Number of Points')
plt.title(f'Break Point Opportunities\n({matches["is_break_point"].mean()*100:.1f}% of points)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("data_intro_break_points.png", dpi=300, bbox_inches='tight')
plt.show()

# 2.1 Rally length distribution
if "rally_count" in matches.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(matches["rally_count"], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(matches["rally_count"].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {matches["rally_count"].mean():.2f}')
    plt.xlabel('Rally Length (number of shots)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rally Lengths', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data_intro_rally_length.png", dpi=300, bbox_inches='tight')
    plt.show()

# 2.2 Serve speed distribution
if "speed_mph" in matches.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(matches["speed_mph"].dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.axvline(matches["speed_mph"].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {matches["speed_mph"].mean():.1f} mph')
    plt.xlabel('Serve Speed (mph)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Serve Speeds', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data_intro_serve_speed.png", dpi=300, bbox_inches='tight')
    plt.show()

# 2.3 Distance run comparison
if "p1_distance_run" in matches.columns and "p2_distance_run" in matches.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(matches["p1_distance_run"].dropna(), bins=30, alpha=0.6, 
             label='Player 1', color='blue', edgecolor='black')
    plt.hist(matches["p2_distance_run"].dropna(), bins=30, alpha=0.6, 
             label='Player 2', color='red', edgecolor='black')
    plt.xlabel('Distance Run (meters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distance Run by Player', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data_intro_distance_run.png", dpi=300, bbox_inches='tight')
    plt.show()

# 2.4 Point win rate by serve status
plt.figure(figsize=(8, 6))
serve_win_data = pd.DataFrame({
    'Serving': ['P1 Serves', 'P2 Serves'],
    'Win Rate': [
        matches[matches["p1_serve"] == 1]["p1_win"].mean(),
        matches[matches["p2_serve"] == 1]["p2_win"].mean()
    ]
})
plt.bar(serve_win_data['Serving'], serve_win_data['Win Rate'], 
        color=['blue', 'red'], edgecolor='black', alpha=0.7)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=2, label='50% (random)')
plt.ylabel('Win Rate')
plt.title('Point Win Rate When Serving', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("data_intro_serve_win_rate.png", dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Correlation Heatmap of Key Predictors
print("\n--- CORRELATION ANALYSIS ---")
corr_vars = [
    "p1_serve", "score_diff", "p1_games_ahead", "p1_sets_ahead",
    "is_break_point", "rally_count", "speed_mph", "p1_win"
]
existing_corr_vars = [v for v in corr_vars if v in matches.columns]

if len(existing_corr_vars) > 1:
    corr_matrix = matches[existing_corr_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                xticklabels=[v.replace('_', ' ').title() for v in existing_corr_vars],
                yticklabels=[v.replace('_', ' ').title() for v in existing_corr_vars])
    plt.title('Correlation Matrix: Key Predictors for Momentum Analysis', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("data_intro_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Save correlation matrix
    corr_matrix.to_csv("data_intro_correlation_matrix.csv")

# 4.1 Win rate by score difference
plt.figure(figsize=(8, 6))
score_win_rate = matches.groupby("score_diff")["p1_win"].mean()
plt.plot(score_win_rate.index, score_win_rate.values, 'o-', linewidth=2, markersize=8)
plt.axhline(0.5, color='red', linestyle='--', linewidth=1, label='50% (random)')
plt.axvline(0, color='gray', linestyle=':', linewidth=1)
plt.xlabel('Score Difference (P1 - P2)')
plt.ylabel('P1 Win Rate')
plt.title('Point Win Rate by Score Difference', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data_intro_win_rate_by_score.png", dpi=300, bbox_inches='tight')
plt.show()

# 4.2 Win rate by games ahead
plt.figure(figsize=(8, 6))
games_win_rate = matches.groupby("p1_games_ahead")["p1_win"].mean()
plt.plot(games_win_rate.index, games_win_rate.values, 'o-', linewidth=2, markersize=8, color='green')
plt.axhline(0.5, color='red', linestyle='--', linewidth=1, label='50% (random)')
plt.axvline(0, color='gray', linestyle=':', linewidth=1)
plt.xlabel('Games Ahead (P1 - P2)')
plt.ylabel('P1 Win Rate')
plt.title('Point Win Rate by Games Ahead', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data_intro_win_rate_by_games.png", dpi=300, bbox_inches='tight')
plt.show()

# 4.3 Win rate on break points vs regular points
plt.figure(figsize=(8, 6))
break_pt_win = matches.groupby("is_break_point")["p1_win"].mean()
plt.bar(["Regular Point", "Break Point"], 
        [break_pt_win.get(0, 0.5), break_pt_win.get(1, 0.5)],
        color=['lightblue', 'red'], edgecolor='black', alpha=0.7)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% (random)')
plt.ylabel('P1 Win Rate')
plt.title('Point Win Rate: Break Points vs Regular Points', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("data_intro_win_rate_break_points.png", dpi=300, bbox_inches='tight')
plt.show()

# 4.4 Win rate by rally length (binned)
if "rally_count" in matches.columns:
    matches["rally_binned"] = pd.cut(matches["rally_count"], bins=[0, 2, 4, 6, 10, 100], 
                                      labels=['1-2', '3-4', '5-6', '7-10', '11+'])
    plt.figure(figsize=(8, 6))
    rally_win_rate = matches.groupby("rally_binned")["p1_win"].mean()
    plt.bar(range(len(rally_win_rate)), rally_win_rate.values, 
            color='purple', edgecolor='black', alpha=0.7)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% (random)')
    plt.xticks(range(len(rally_win_rate)), rally_win_rate.index)
    plt.xlabel('Rally Length (shots)')
    plt.ylabel('P1 Win Rate')
    plt.title('Point Win Rate by Rally Length', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("data_intro_win_rate_by_rally.png", dpi=300, bbox_inches='tight')
    plt.show()

print("\n--- DATA INTRODUCTION VISUALIZATIONS SAVED ---")
print("Saved files:")
print("  - data_intro_serving_advantage.png")
print("  - data_intro_score_difference.png")
print("  - data_intro_games_ahead.png")
print("  - data_intro_break_points.png")
print("  - data_intro_rally_length.png")
print("  - data_intro_serve_speed.png")
print("  - data_intro_distance_run.png")
print("  - data_intro_serve_win_rate.png")
print("  - data_intro_correlation_heatmap.png")
print("  - data_intro_win_rate_by_score.png")
print("  - data_intro_win_rate_by_games.png")
print("  - data_intro_win_rate_break_points.png")
print("  - data_intro_win_rate_by_rally.png")
print("  - data_intro_summary_stats.csv")
print("  - data_intro_correlation_matrix.csv")


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







# --------------------------------------------------------------
# STATISTICAL SIGNIFICANCE TESTING: Is Momentum Real?
# --------------------------------------------------------------

from scipy.stats import ttest_ind, ttest_1samp

print("\n--- STATISTICAL SIGNIFICANCE TESTING ---")

# Prepare data for significance testing
# We need matches with previous point data
test_matches = matches.dropna(subset=["prev_point_victor"]).copy()

# For each match, test if momentum is significant
momentum_results = []

for match_id in test_matches["match_id"].unique():
    match_data = test_matches[test_matches["match_id"] == match_id].copy()
    
    if len(match_data) < 10:  # Skip matches with too few points
        continue
    
    # Player 1 momentum test
    p1_after_win = match_data[match_data["p1_prev_win"] == 1]["p1_win"].values
    p1_after_loss = match_data[match_data["p1_prev_win"] == 0]["p1_win"].values
    
    if len(p1_after_win) > 0 and len(p1_after_loss) > 0:
        # T-test: Is P(win|won previous) different from P(win|lost previous)?
        tstat_p1, pval_p1 = ttest_ind(p1_after_win, p1_after_loss)
        p1_significant = 1 if pval_p1 < 0.05 else 0
    else:
        pval_p1 = 1.0
        p1_significant = 0
    
    # Player 2 momentum test
    p2_after_win = match_data[match_data["p2_prev_win"] == 1]["p2_win"].values
    p2_after_loss = match_data[match_data["p2_prev_win"] == 0]["p2_win"].values
    
    if len(p2_after_win) > 0 and len(p2_after_loss) > 0:
        tstat_p2, pval_p2 = ttest_ind(p2_after_win, p2_after_loss)
        p2_significant = 1 if pval_p2 < 0.05 else 0
    else:
        pval_p2 = 1.0
        p2_significant = 0
    
    # Calculate turning points (where momentum shifts)
    # A turning point is where the player with higher recent wins switches
    match_data_sorted = match_data.sort_values(["set_no", "game_no", "point_no"]).reset_index(drop=True)
    
    # Use rolling window to identify momentum shifts
    window = 5
    p1_recent_wins = match_data_sorted["p1_win"].rolling(window=window, min_periods=1).sum()
    p2_recent_wins = match_data_sorted["p2_win"].rolling(window=window, min_periods=1).sum()
    
    # Identify turning points (where leader switches)
    leader = (p1_recent_wins > p2_recent_wins).astype(int)
    turning_points = (leader.diff() != 0).sum() - 1  # Subtract 1 for initial diff
    turning_points = max(0, turning_points)
    
    # Test if turning points are significant (compare to random expectation)
    # Expected turning points in random sequence
    n_points = len(match_data_sorted)
    expected_turning_points = n_points / window  # Rough estimate
    
    if turning_points > 0 and expected_turning_points > 0:
        # Z-test approximation for turning point significance
        z_score = (turning_points - expected_turning_points) / np.sqrt(expected_turning_points)
        p1_turning_significant = 1 if abs(z_score) > 1.96 else 0  # 95% confidence
    else:
        p1_turning_significant = 0
    
    # Same for player 2 (symmetric)
    p2_turning_significant = p1_turning_significant
    
    momentum_results.append({
        "match_id": match_id,
        "p1_momentum_pval": pval_p1,
        "p1_momentum_significant": p1_significant,
        "p2_momentum_pval": pval_p2,
        "p2_momentum_significant": p2_significant,
        "p1_turning_significant": p1_turning_significant,
        "p2_turning_significant": p2_turning_significant,
        "turning_points": turning_points
    })

momentum_df = pd.DataFrame(momentum_results)

print("\n--- MOMENTUM SIGNIFICANCE RESULTS ---")
print(f"Total matches analyzed: {len(momentum_df)}")
print(f"\nPlayer 1 Momentum:")
print(f"  Significant matches: {momentum_df['p1_momentum_significant'].sum()} ({100*momentum_df['p1_momentum_significant'].mean():.1f}%)")
print(f"  Mean p-value: {momentum_df['p1_momentum_pval'].mean():.4f}")

print(f"\nPlayer 2 Momentum:")
print(f"  Significant matches: {momentum_df['p2_momentum_significant'].sum()} ({100*momentum_df['p2_momentum_significant'].mean():.1f}%)")
print(f"  Mean p-value: {momentum_df['p2_momentum_pval'].mean():.4f}")

print(f"\nTurning Points:")
print(f"  Player 1 significant: {momentum_df['p1_turning_significant'].sum()} ({100*momentum_df['p1_turning_significant'].mean():.1f}%)")
print(f"  Player 2 significant: {momentum_df['p2_turning_significant'].sum()} ({100*momentum_df['p2_turning_significant'].mean():.1f}%)")
print(f"  Mean turning points per match: {momentum_df['turning_points'].mean():.2f}")

# Summary statistics table
summary_stats = pd.DataFrame({
    "Statistic": ["P1 Momentum Significant", "P2 Momentum Significant", 
                  "P1 Turning Significant", "P2 Turning Significant"],
    "Mean": [
        momentum_df['p1_momentum_significant'].mean(),
        momentum_df['p2_momentum_significant'].mean(),
        momentum_df['p1_turning_significant'].mean(),
        momentum_df['p2_turning_significant'].mean()
    ],
    "Std": [
        momentum_df['p1_momentum_significant'].std(),
        momentum_df['p2_momentum_significant'].std(),
        momentum_df['p1_turning_significant'].std(),
        momentum_df['p2_turning_significant'].std()
    ]
})

print("\n--- SUMMARY STATISTICS TABLE ---")
print(summary_stats.to_string(index=False))

# Save results for report
momentum_df.to_csv("momentum_significance_results.csv", index=False)
summary_stats.to_csv("momentum_summary_stats.csv", index=False)


# --------------------------------------------------------------
# 4️⃣ OPTIMIZED MODEL: TRAIN ON NON-FINALS, TEST ON FINALS
# --------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Identify test matches: semi-finals (1601, 1602) and final (1701)
# Based on match_id format: "2023-wimbledon-XXXX" where XXXX indicates round and match
# Semi-finals: round 6, matches 01-02 = "1601", "1602"
# Final: round 7, match 01 = "1701"
test_match_ids = ['2023-wimbledon-1601', '2023-wimbledon-1602', '2023-wimbledon-1701']
test_match_ids = [m for m in test_match_ids if m in matches['match_id'].values]

print("\n--- TEST MATCH IDENTIFICATION ---")
print(f"Test match IDs: {test_match_ids}")
print(f"Number of test matches: {len(test_match_ids)}")

# Split data: training = all matches except test matches, test = test matches
train_data = matches[~matches['match_id'].isin(test_match_ids)].copy()
test_data = matches[matches['match_id'].isin(test_match_ids)].copy()

print(f"\nTraining set size: {len(train_data)} points")
print(f"Test set size: {len(test_data)} points")

# --------------------------------------------------------------
# FEATURE ENGINEERING: Create comprehensive momentum features
# --------------------------------------------------------------

def create_momentum_features(df):
    """
    Create comprehensive features for momentum prediction including:
    - Previous point outcomes
    - Streaks
    - Game/set context
    - Player performance metrics
    """
    df = df.copy()
    
    # Sort by match and point order
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)
    
    # Group by match for match-specific calculations
    grouped = df.groupby("match_id")
    
    # 1. Previous point outcomes (momentum indicators)
    df["prev_point_victor"] = grouped["point_victor"].shift(1)
    df["p1_prev_win"] = (df["prev_point_victor"] == 1).astype(int)
    df["p2_prev_win"] = (df["prev_point_victor"] == 2).astype(int)
    
    # 2. Win streaks for each player
    # IMPORTANT: Use shifted wins to avoid data leakage (only use previous points)
    df["p1_win"] = (df["point_victor"] == 1).astype(int)
    df["p2_win"] = (df["point_victor"] == 2).astype(int)
    
    # Shift wins to exclude current point (only use previous points for streaks)
    df["p1_win_prev"] = grouped["p1_win"].shift(1).fillna(0).astype(int)
    df["p2_win_prev"] = grouped["p2_win"].shift(1).fillna(0).astype(int)
    
    def compute_streaks_p1(x):
        streaks = []
        current = 0
        for val in (x == 1):
            if val:
                current += 1
            else:
                current = 0
            streaks.append(current)
        return streaks
    
    def compute_streaks_p2(x):
        streaks = []
        current = 0
        for val in (x == 2):
            if val:
                current += 1
            else:
                current = 0
            streaks.append(current)
        return streaks
    
    # Compute streaks using PREVIOUS wins only (no data leakage)
    df["p1_streak"] = grouped["p1_win_prev"].transform(compute_streaks_p1)
    df["p2_streak"] = grouped["p2_win_prev"].transform(compute_streaks_p2)
    
    # 3. Recent performance (last 3, 5 points)
    df["p1_wins_last3"] = grouped["p1_win"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    df["p1_wins_last5"] = grouped["p1_win"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    
    # 4. Serve advantage
    df["p1_serve"] = (df["server"] == 1).astype(int)
    df["p2_serve"] = (df["server"] == 2).astype(int)
    # REMOVED server_won - it uses current point_victor (data leakage)
    # Instead, use historical server win rate (rolling average of previous points)
    # Calculate server_won for previous points only, then take rolling average
    df["server_won_temp"] = (df["server"] == df["point_victor"]).astype(int)
    df["server_won_prev"] = grouped["server_won_temp"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.5)
    df = df.drop(columns=["server_won_temp"])  # Remove temporary column
    
    # 5. Game context (score within game)
    df["p1_score_numeric"] = df["p1_score"].map({"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}).fillna(0)
    df["p2_score_numeric"] = df["p2_score"].map({"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}).fillna(0)
    df["score_diff"] = df["p1_score_numeric"] - df["p2_score_numeric"]
    
    # 6. Set context
    df["p1_games_ahead"] = df["p1_games"] - df["p2_games"]
    df["p1_sets_ahead"] = df["p1_sets"] - df["p2_sets"]
    
    # 7. Break point context
    df["p1_break_pt"] = df["p1_break_pt"].fillna(0).astype(int)
    df["p2_break_pt"] = df["p2_break_pt"].fillna(0).astype(int)
    
    # 8. Performance metrics (rolling averages) - using shifted values
    df["p1_avg_rally"] = grouped["rally_count"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p1_avg_distance"] = grouped["p1_distance_run"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    
    # 9. Serve quality indicators - serve speed is OK (known before point outcome)
    df["serve_speed"] = df["speed_mph"].fillna(df["speed_mph"].median())
    
    # 10. Error rates (rolling) - using shifted values
    df["p1_error_rate"] = grouped["p1_unf_err"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p2_error_rate"] = grouped["p2_unf_err"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    
    # 11. Cumulative totals - shift to exclude current point
    df["p1_points_won_prev"] = grouped["p1_points_won"].shift(1).fillna(0).astype(int)
    df["p2_points_won_prev"] = grouped["p2_points_won"].shift(1).fillna(0).astype(int)
    
    # 12. Historical performance metrics (shifted to avoid data leakage)
    df["p1_ace_rate"] = grouped["p1_ace"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p2_ace_rate"] = grouped["p2_ace"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p1_winner_rate"] = grouped["p1_winner"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p2_winner_rate"] = grouped["p2_winner"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p1_double_fault_rate"] = grouped["p1_double_fault"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p2_double_fault_rate"] = grouped["p2_double_fault"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p1_net_pt_won_rate"] = grouped["p1_net_pt_won"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df["p2_net_pt_won_rate"] = grouped["p2_net_pt_won"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    
    return df

# --------------------------------------------------------------
# MOMENTUM QUANTIFICATION: Create unified momentum score
# --------------------------------------------------------------

def calculate_momentum_score(df, feature_weights=None):
    """
    Calculate a unified momentum score (0-1 scale) for each player at each point.
    Uses weighted combination of key momentum features.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with momentum features already created
    feature_weights : dict, optional
        Custom weights for features. If None, uses default weights.
    
    Returns:
    --------
    DataFrame with p1_momentum, p2_momentum, and momentum_diff columns added
    """
    df = df.copy()
    
    # Default feature weights (can be adjusted based on feature importance)
    if feature_weights is None:
        feature_weights = {
            'streak': 0.25,           # Current win streak
            'recent_wins': 0.20,      # Recent performance (last 3-5 points)
            'game_context': 0.15,     # Score difference, games ahead
            'serve_advantage': 0.15,  # Serving advantage
            'performance': 0.15,      # Rally length, distance, errors
            'break_point': 0.10       # Break point opportunities
        }
    
    # Normalize features to 0-1 scale for each match
    grouped = df.groupby("match_id")
    
    # Player 1 momentum components
    # 1. Streak component (normalized by max streak in match)
    p1_max_streak = grouped["p1_streak"].transform("max")
    p1_streak_norm = df["p1_streak"] / (p1_max_streak + 1)  # +1 to avoid division by zero
    
    # 2. Recent wins component (normalized)
    p1_recent_norm = df["p1_wins_last5"] / 5.0  # Max is 5
    
    # 3. Game context component (score difference normalized)
    max_score_diff = grouped["score_diff"].transform(lambda x: x.abs().max())
    p1_game_context = (df["score_diff"] + max_score_diff) / (2 * max_score_diff + 1)
    p1_game_context = p1_game_context.fillna(0.5)
    
    # Games ahead (normalized)
    max_games_ahead = grouped["p1_games_ahead"].transform(lambda x: x.abs().max())
    p1_games_norm = (df["p1_games_ahead"] + max_games_ahead) / (2 * max_games_ahead + 1)
    p1_games_norm = p1_games_norm.fillna(0.5)
    
    # Sets ahead (normalized)
    max_sets_ahead = grouped["p1_sets_ahead"].transform(lambda x: x.abs().max())
    p1_sets_norm = (df["p1_sets_ahead"] + max_sets_ahead) / (2 * max_sets_ahead + 1)
    p1_sets_norm = p1_sets_norm.fillna(0.5)
    
    game_context_combined = (p1_game_context + p1_games_norm + p1_sets_norm) / 3
    
    # 4. Serve advantage component
    p1_serve_adv = df["p1_serve"].astype(float)  # 1 if serving, 0 otherwise
    
    # 5. Performance component (inverse of errors, normalized rally performance)
    p1_error_norm = 1 - df["p1_error_rate"].fillna(0)  # Lower errors = higher momentum
    p2_error_norm = df["p2_error_rate"].fillna(0)  # Opponent errors help momentum
    p1_perf_component = (p1_error_norm + p2_error_norm) / 2
    
    # 6. Break point component
    p1_break_pt = df["p1_break_pt"].astype(float)
    p2_break_pt = df["p2_break_pt"].astype(float)
    p1_break_component = p1_break_pt * 0.7 - p2_break_pt * 0.3  # Having break point helps, opponent having it hurts
    p1_break_component = (p1_break_component + 1) / 2  # Normalize to 0-1
    
    # Combine all components for Player 1
    df["p1_momentum"] = (
        feature_weights['streak'] * p1_streak_norm.fillna(0) +
        feature_weights['recent_wins'] * p1_recent_norm.fillna(0) +
        feature_weights['game_context'] * game_context_combined.fillna(0.5) +
        feature_weights['serve_advantage'] * p1_serve_adv.fillna(0.5) +
        feature_weights['performance'] * p1_perf_component.fillna(0.5) +
        feature_weights['break_point'] * p1_break_component.fillna(0.5)
    )
    
    # Ensure momentum is between 0 and 1
    df["p1_momentum"] = df["p1_momentum"].clip(0, 1)
    
    # Player 2 momentum (symmetric calculation)
    p2_max_streak = grouped["p2_streak"].transform("max")
    p2_streak_norm = df["p2_streak"] / (p2_max_streak + 1)
    
    p2_recent_norm = (5 - df["p1_wins_last5"]) / 5.0  # Inverse of p1 wins
    
    p2_game_context = (-df["score_diff"] + max_score_diff) / (2 * max_score_diff + 1)
    p2_game_context = p2_game_context.fillna(0.5)
    p2_games_norm = (-df["p1_games_ahead"] + max_games_ahead) / (2 * max_games_ahead + 1)
    p2_games_norm = p2_games_norm.fillna(0.5)
    p2_sets_norm = (-df["p1_sets_ahead"] + max_sets_ahead) / (2 * max_sets_ahead + 1)
    p2_sets_norm = p2_sets_norm.fillna(0.5)
    
    game_context_combined_p2 = (p2_game_context + p2_games_norm + p2_sets_norm) / 3
    
    p2_serve_adv = df["p2_serve"].astype(float)
    
    p2_error_norm = 1 - df["p2_error_rate"].fillna(0)
    p1_error_norm_p2 = df["p1_error_rate"].fillna(0)
    p2_perf_component = (p2_error_norm + p1_error_norm_p2) / 2
    
    p2_break_component = p2_break_pt * 0.7 - p1_break_pt * 0.3
    p2_break_component = (p2_break_component + 1) / 2
    
    df["p2_momentum"] = (
        feature_weights['streak'] * p2_streak_norm.fillna(0) +
        feature_weights['recent_wins'] * p2_recent_norm.fillna(0) +
        feature_weights['game_context'] * game_context_combined_p2.fillna(0.5) +
        feature_weights['serve_advantage'] * p2_serve_adv.fillna(0.5) +
        feature_weights['performance'] * p2_perf_component.fillna(0.5) +
        feature_weights['break_point'] * p2_break_component.fillna(0.5)
    )
    
    df["p2_momentum"] = df["p2_momentum"].clip(0, 1)
    
    # Momentum difference (positive = P1 ahead, negative = P2 ahead)
    df["momentum_diff"] = df["p1_momentum"] - df["p2_momentum"]
    
    return df

# Apply feature engineering to both train and test sets
print("\n--- CREATING FEATURES ---")
train_features = create_momentum_features(train_data)
test_features = create_momentum_features(test_data)

# Calculate momentum scores
print("\n--- CALCULATING MOMENTUM SCORES ---")
train_features = calculate_momentum_score(train_features)
test_features = calculate_momentum_score(test_features)

print(f"Momentum score statistics:")
print(f"  P1 momentum - Mean: {train_features['p1_momentum'].mean():.3f}, Std: {train_features['p1_momentum'].std():.3f}")
print(f"  P2 momentum - Mean: {train_features['p2_momentum'].mean():.3f}, Std: {train_features['p2_momentum'].std():.3f}")
print(f"  Momentum diff - Mean: {train_features['momentum_diff'].mean():.3f}, Std: {train_features['momentum_diff'].std():.3f}")

# --------------------------------------------------------------
# SELECT FEATURES FOR MODELING
# --------------------------------------------------------------

# Select relevant features (excluding identifiers and target)
# REMOVED features that cause data leakage (current point outcomes)
feature_cols = [
    # Momentum features
    "p1_prev_win", "p2_prev_win",
    "p1_streak", "p2_streak",
    "p1_wins_last3", "p1_wins_last5",
    
    # Serve and context
    "p1_serve",
    "server_won_prev",  # Historical server win rate (shifted)
    "serve_no",
    
    # Game/set context (these are OK - state before point)
    "score_diff", "p1_games_ahead", "p1_sets_ahead",
    "p1_break_pt", "p2_break_pt",
    
    # Performance metrics (using shifted/rolling averages)
    "p1_avg_rally", "p1_avg_distance",
    "serve_speed",  # OK - serve speed known before point outcome
    "p1_error_rate", "p2_error_rate",
    
    # Historical player stats (shifted cumulative totals and rolling rates)
    "p1_points_won_prev", "p2_points_won_prev",
    "p1_ace_rate", "p2_ace_rate",
    "p1_winner_rate", "p2_winner_rate",
    "p1_double_fault_rate", "p2_double_fault_rate",
    "p1_net_pt_won_rate", "p2_net_pt_won_rate",
    
    # REMOVED (data leakage - current point outcomes):
    # "server_won", "rally_count", "p1_distance_run", "p2_distance_run",
    # "p1_points_won", "p2_points_won", "p1_ace", "p2_ace",
    # "p1_winner", "p2_winner", "p1_double_fault", "p2_double_fault",
    # "p1_unf_err", "p2_unf_err", "p1_net_pt_won", "p2_net_pt_won"
]

# Only use features that exist in the dataframe
available_features = [f for f in feature_cols if f in train_features.columns]

# Target: predict if player 1 wins the point
target_col = "p1_win"

# Prepare training data
train_clean = train_features.dropna(subset=[target_col] + available_features)
X_train = train_clean[available_features]
y_train = train_clean[target_col]

# Prepare test data
test_clean = test_features.dropna(subset=[target_col] + available_features)
X_test = test_clean[available_features]
y_test = test_clean[target_col]

print(f"\nFeatures used: {len(available_features)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# --------------------------------------------------------------
# TRAIN OPTIMIZED MODEL
# --------------------------------------------------------------

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try both Random Forest and Logistic Regression, choose best
print("\n--- TRAINING MODELS ---")

# Model 1: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Model 2: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Choose best model
if rf_accuracy >= lr_accuracy:
    model = rf_model
    model_name = "Random Forest"
    print(f"Selected: Random Forest (accuracy: {rf_accuracy:.4f} vs Logistic Regression: {lr_accuracy:.4f})")
else:
    model = lr_model
    model_name = "Logistic Regression"
    print(f"Selected: Logistic Regression (accuracy: {lr_accuracy:.4f} vs Random Forest: {rf_accuracy:.4f})")

# Get feature importance (works for RF, coefficients for LR)
if model_name == "Random Forest":
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)


# --------------------------------------------------------------
# EVALUATE ON TEST SET (FINAL MATCH)
# --------------------------------------------------------------

print("\n--- TOP 10 MOST IMPORTANT FEATURES ---")
if model_name == "Random Forest":
    print(feature_importance.head(10))
else:
    print(feature_importance.head(10))
    print("\n(Shown as absolute coefficient values)")

print("\n--- MODEL EVALUATION ON FINAL MATCH ---")
print(f"Using model: {model_name}")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on final match: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Player 2 wins", "Player 1 wins"]))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Additional metrics
from sklearn.metrics import roc_auc_score, log_loss
try:
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")
except:
    print("\nROC-AUC Score: N/A (only one class in test set)")

logloss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {logloss:.4f}")

# --------------------------------------------------------------
# COMPARISON: BASELINE vs MODEL
# --------------------------------------------------------------

# Baseline: always predict server wins
baseline_pred = (test_clean["p1_serve"] == 1).astype(int)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
print(f"\n--- BASELINE COMPARISON ---")
print(f"Baseline (server always wins) accuracy: {baseline_accuracy:.4f}")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Improvement: {accuracy - baseline_accuracy:.4f}")

# --------------------------------------------------------------
# VISUALIZATION: Predictions vs Actual
# --------------------------------------------------------------

# Plot predictions over the course of the final match
if len(test_clean) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Actual vs Predicted over points
    match_points = test_clean.reset_index(drop=True)
    axes[0].plot(range(len(y_test)), y_test.values, 'o-', label='Actual (P1 wins)', alpha=0.7, markersize=3)
    axes[0].plot(range(len(y_pred)), y_pred, 's-', label='Predicted (P1 wins)', alpha=0.7, markersize=3)
    axes[0].set_xlabel('Point Number')
    axes[0].set_ylabel('Player 1 Wins Point')
    axes[0].set_title('Actual vs Predicted Point Outcomes (Final Match)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction probabilities
    axes[1].plot(range(len(y_pred_proba)), y_pred_proba, 'g-', alpha=0.7, linewidth=1)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    axes[1].set_xlabel('Point Number')
    axes[1].set_ylabel('Predicted Probability (P1 wins)')
    axes[1].set_title('Prediction Probabilities Over Match')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n--- MODEL TRAINING AND EVALUATION COMPLETE ---")


# --------------------------------------------------------------
# MOMENTUM SWING PREDICTION MODEL
# --------------------------------------------------------------

print("\n--- MOMENTUM SWING PREDICTION MODEL ---")
print("Predicting when momentum swings will occur in the next N points")

def create_swing_target(df, lookahead=5, swing_threshold=0.15):
    """
    Create target variable for momentum swing prediction.
    A swing occurs when momentum_diff changes sign or changes by more than threshold.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with momentum_diff already calculated
    lookahead : int
        Number of points ahead to check for swing
    swing_threshold : float
        Minimum change in momentum_diff to count as swing
    
    Returns:
    --------
    DataFrame with 'will_swing_next_N' column added
    """
    df = df.copy()
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)
    
    grouped = df.groupby("match_id")
    
    # Calculate future momentum changes
    df["momentum_diff_future"] = grouped["momentum_diff"].shift(-lookahead)
    df["momentum_diff_current"] = df["momentum_diff"]
    
    # A swing occurs if:
    # 1. Momentum diff changes sign (crosses zero), OR
    # 2. Absolute change exceeds threshold
    momentum_change = df["momentum_diff_future"] - df["momentum_diff_current"]
    sign_change = (df["momentum_diff_current"] > 0) != (df["momentum_diff_future"] > 0)
    large_change = momentum_change.abs() > swing_threshold
    
    # Target: will swing occur in next N points?
    df["will_swing_next_N"] = (sign_change | large_change).astype(int)
    
    # Fill NaN values (last N points of each match) with 0
    df["will_swing_next_N"] = df["will_swing_next_N"].fillna(0).astype(int)
    
    return df

# Create swing targets for training data
print("\n--- CREATING SWING TARGETS ---")
train_features_swing = create_swing_target(train_features.copy(), lookahead=5, swing_threshold=0.15)
test_features_swing = create_swing_target(test_features.copy(), lookahead=5, swing_threshold=0.15)

# Prepare features for swing prediction (use same features as point prediction)
swing_feature_cols = available_features.copy()

# Prepare training data for swing prediction
train_swing_clean = train_features_swing.dropna(subset=["will_swing_next_N"] + swing_feature_cols)
X_swing_train = train_swing_clean[swing_feature_cols]
y_swing_train = train_swing_clean["will_swing_next_N"]

# Prepare test data
test_swing_clean = test_features_swing.dropna(subset=["will_swing_next_N"] + swing_feature_cols)
X_swing_test = test_swing_clean[swing_feature_cols]
y_swing_test = test_swing_clean["will_swing_next_N"]

print(f"\nSwing prediction features: {len(swing_feature_cols)}")
print(f"Training samples: {len(X_swing_train)}")
print(f"Test samples: {len(X_swing_test)}")
print(f"Swing rate in training: {y_swing_train.mean():.3f}")
print(f"Swing rate in test: {y_swing_test.mean():.3f}")

# Standardize features
scaler_swing = StandardScaler()
X_swing_train_scaled = scaler_swing.fit_transform(X_swing_train)
X_swing_test_scaled = scaler_swing.transform(X_swing_test)

# Train swing prediction models
print("\n--- TRAINING SWING PREDICTION MODELS ---")

# Model 1: Random Forest
rf_swing_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)
rf_swing_model.fit(X_swing_train_scaled, y_swing_train)
rf_swing_pred = rf_swing_model.predict(X_swing_test_scaled)
rf_swing_proba = rf_swing_model.predict_proba(X_swing_test_scaled)[:, 1]
rf_swing_accuracy = accuracy_score(y_swing_test, rf_swing_pred)

# Model 2: Logistic Regression
lr_swing_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_swing_model.fit(X_swing_train_scaled, y_swing_train)
lr_swing_pred = lr_swing_model.predict(X_swing_test_scaled)
lr_swing_proba = lr_swing_model.predict_proba(X_swing_test_scaled)[:, 1]
lr_swing_accuracy = accuracy_score(y_swing_test, lr_swing_pred)

# Choose best model
if rf_swing_accuracy >= lr_swing_accuracy:
    swing_model = rf_swing_model
    swing_model_name = "Random Forest"
    swing_pred = rf_swing_pred
    swing_proba = rf_swing_proba
    print(f"Selected: Random Forest (accuracy: {rf_swing_accuracy:.4f} vs Logistic Regression: {lr_swing_accuracy:.4f})")
else:
    swing_model = lr_swing_model
    swing_model_name = "Logistic Regression"
    swing_pred = lr_swing_pred
    swing_proba = lr_swing_proba
    print(f"Selected: Logistic Regression (accuracy: {lr_swing_accuracy:.4f} vs Random Forest: {rf_swing_accuracy:.4f})")

# Evaluate swing prediction
swing_accuracy = accuracy_score(y_swing_test, swing_pred)
print(f"\n--- SWING PREDICTION EVALUATION ---")
print(f"Model: {swing_model_name}")
print(f"Accuracy: {swing_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_swing_test, swing_pred, target_names=["No Swing", "Swing"]))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_swing_test, swing_pred))

# Additional metrics
try:
    swing_auc = roc_auc_score(y_swing_test, swing_proba)
    print(f"\nROC-AUC Score: {swing_auc:.4f}")
except:
    print("\nROC-AUC Score: N/A")

swing_logloss = log_loss(y_swing_test, swing_proba)
print(f"Log Loss: {swing_logloss:.4f}")

# Feature importance for swing prediction
if swing_model_name == "Random Forest":
    swing_feature_importance = pd.DataFrame({
        'feature': swing_feature_cols,
        'importance': swing_model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    swing_feature_importance = pd.DataFrame({
        'feature': swing_feature_cols,
        'coefficient': swing_model.coef_[0],
        'abs_coefficient': np.abs(swing_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

print("\n--- TOP 10 FEATURES FOR SWING PREDICTION ---")
print(swing_feature_importance.head(10))

# Save swing predictions back to test data
test_swing_clean["predicted_swing"] = swing_pred
test_swing_clean["predicted_swing_proba"] = swing_proba
test_swing_clean["actual_swing"] = y_swing_test.values

# Analyze swing prediction by match
print("\n--- SWING PREDICTION BY MATCH ---")
swing_by_match = []
for match_id in test_match_ids:
    match_swing_data = test_swing_clean[test_swing_clean["match_id"] == match_id]
    if len(match_swing_data) == 0:
        continue
    
    match_swing_accuracy = accuracy_score(
        match_swing_data["actual_swing"],
        match_swing_data["predicted_swing"]
    )
    
    actual_swings = match_swing_data["actual_swing"].sum()
    predicted_swings = match_swing_data["predicted_swing"].sum()
    total_points = len(match_swing_data)
    
    swing_by_match.append({
        "match_id": match_id,
        "total_points": total_points,
        "actual_swings": actual_swings,
        "predicted_swings": predicted_swings,
        "swing_rate_actual": actual_swings / total_points,
        "swing_rate_predicted": predicted_swings / total_points,
        "accuracy": match_swing_accuracy
    })

swing_by_match_df = pd.DataFrame(swing_by_match)
print(swing_by_match_df.to_string(index=False))

# Save swing analysis
swing_by_match_df.to_csv("swing_prediction_by_match.csv", index=False)
swing_feature_importance.to_csv("swing_feature_importance.csv", index=False)

print("\n--- SWING PREDICTION ANALYSIS COMPLETE ---")


# --------------------------------------------------------------
# SWING PREDICTION VISUALIZATIONS
# --------------------------------------------------------------

print("\n--- CREATING SWING PREDICTION VISUALIZATIONS ---")

# Prepare data for visualization
test_swing_sorted = test_swing_clean.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

# Create a comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Swing predictions over time for each match (subplot grid)
for idx, match_id in enumerate(test_match_ids):
    match_swing_data = test_swing_sorted[test_swing_sorted["match_id"] == match_id].copy()
    
    if len(match_swing_data) == 0:
        continue
    
    # Determine subplot position
    row = idx // 2
    col = idx % 2
    
    ax = fig.add_subplot(gs[row, col])
    
    point_numbers = range(len(match_swing_data))
    
    # Plot swing prediction probabilities
    ax.plot(point_numbers, match_swing_data["predicted_swing_proba"].values, 
            'b-', alpha=0.6, linewidth=1.5, label='Predicted Swing Probability')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Decision Threshold')
    
    # Mark actual swings
    actual_swing_points = match_swing_data[match_swing_data["actual_swing"] == 1].index
    if len(actual_swing_points) > 0:
        ax.scatter(actual_swing_points, 
                  match_swing_data.loc[actual_swing_points, "predicted_swing_proba"].values,
                  color='green', marker='o', s=50, alpha=0.7, zorder=5, 
                  label='Actual Swing', edgecolors='darkgreen', linewidths=1)
    
    # Mark predicted swings
    predicted_swing_points = match_swing_data[match_swing_data["predicted_swing"] == 1].index
    if len(predicted_swing_points) > 0:
        ax.scatter(predicted_swing_points,
                  match_swing_data.loc[predicted_swing_points, "predicted_swing_proba"].values,
                  color='red', marker='x', s=40, alpha=0.7, zorder=5,
                  label='Predicted Swing', linewidths=2)
    
    # Mark set boundaries
    set_boundaries = []
    current_set = None
    for i, set_num in enumerate(match_swing_data["set_no"]):
        if current_set is None:
            current_set = set_num
        elif set_num != current_set:
            set_boundaries.append(i)
            current_set = set_num
            ax.axvline(x=i, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Point Number')
    ax.set_ylabel('Swing Probability')
    ax.set_title(f'{match_id}\nActual: {match_swing_data["actual_swing"].sum()}, Predicted: {match_swing_data["predicted_swing"].sum()}, Accuracy: {accuracy_score(match_swing_data["actual_swing"], match_swing_data["predicted_swing"]):.3f}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

# 2. Swing prediction accuracy comparison across matches
ax2 = fig.add_subplot(gs[2, 0])
match_names = [m.split('-')[-1] for m in test_match_ids]  # Extract match numbers
accuracies = swing_by_match_df["accuracy"].values
colors_bar = ['green' if acc > 0.6 else 'orange' if acc > 0.5 else 'red' for acc in accuracies]
bars = ax2.bar(match_names, accuracies, color=colors_bar, alpha=0.7, edgecolor='black')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (50%)')
ax2.set_ylabel('Swing Prediction Accuracy')
ax2.set_xlabel('Match ID')
ax2.set_title('Swing Prediction Accuracy by Match')
ax2.set_ylim([0, 1])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Actual vs Predicted swing rates
ax3 = fig.add_subplot(gs[2, 1])
x_pos = np.arange(len(match_names))
width = 0.35
bars1 = ax3.bar(x_pos - width/2, swing_by_match_df["swing_rate_actual"].values, 
                width, label='Actual Swing Rate', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x_pos + width/2, swing_by_match_df["swing_rate_predicted"].values, 
                width, label='Predicted Swing Rate', color='coral', alpha=0.7, edgecolor='black')
ax3.set_ylabel('Swing Rate')
ax3.set_xlabel('Match ID')
ax3.set_title('Actual vs Predicted Swing Rates')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(match_names)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=9)

plt.suptitle('Momentum Swing Prediction Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig("swing_prediction_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Feature importance for swing prediction
fig2, ax4 = plt.subplots(figsize=(12, 8))
top_swing_features = swing_feature_importance.head(15)
if swing_model_name == "Random Forest":
    ax4.barh(range(len(top_swing_features)), top_swing_features["importance"].values, 
             color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Feature Importance')
else:
    ax4.barh(range(len(top_swing_features)), top_swing_features["abs_coefficient"].values, 
             color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Absolute Coefficient Value')
ax4.set_yticks(range(len(top_swing_features)))
ax4.set_yticklabels(top_swing_features["feature"].values)
ax4.set_title(f'Top 15 Features for Swing Prediction ({swing_model_name})', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("swing_feature_importance_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Confusion matrix visualization for swing prediction
fig3, ax5 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_swing_test, swing_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['No Swing', 'Swing'],
            yticklabels=['No Swing', 'Swing'],
            cbar_kws={'label': 'Count'})
ax5.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax5.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax5.set_title('Swing Prediction Confusion Matrix', fontsize=14, fontweight='bold')

# Add accuracy text
total = cm.sum()
correct = cm[0, 0] + cm[1, 1]
accuracy_text = f'Accuracy: {correct/total:.3f}'
ax5.text(0.5, -0.15, accuracy_text, transform=ax5.transAxes,
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("swing_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n--- SWING PREDICTION VISUALIZATIONS SAVED ---")
print("Saved files:")
print("  - swing_prediction_analysis.png (comprehensive analysis)")
print("  - swing_feature_importance_plot.png (feature importance)")
print("  - swing_confusion_matrix.png (confusion matrix)")


# --------------------------------------------------------------
# SET-BY-SET ANALYSIS OF TEST MATCHES
# --------------------------------------------------------------

print("\n--- SET-BY-SET ANALYSIS OF TEST MATCHES ---")

# Merge predictions back into test_clean for set analysis
test_clean_with_pred = test_clean.copy()
test_clean_with_pred["predicted_p1_win"] = y_pred
test_clean_with_pred["predicted_proba"] = y_pred_proba
test_clean_with_pred["actual_p1_win"] = y_test.values

# Add swing predictions
test_clean_with_pred = test_clean_with_pred.merge(
    test_swing_clean[["match_id", "set_no", "game_no", "point_no", 
                      "predicted_swing", "actual_swing", "predicted_swing_proba"]],
    on=["match_id", "set_no", "game_no", "point_no"],
    how="left"
)

# Analyze by set and match
set_analysis = []

for match_id in test_match_ids:
    match_data = test_clean_with_pred[test_clean_with_pred["match_id"] == match_id].copy()
    
    if len(match_data) == 0:
        continue
    
    for set_num in sorted(match_data["set_no"].unique()):
        set_data = match_data[match_data["set_no"] == set_num].copy()
        
        if len(set_data) == 0:
            continue
        
        # Calculate metrics for this set
        total_points = len(set_data)
        p1_wins_actual = set_data["actual_p1_win"].sum()
        p2_wins_actual = total_points - p1_wins_actual
        p1_wins_pred = set_data["predicted_p1_win"].sum()
        p2_wins_pred = total_points - p1_wins_pred
        
        # Accuracy for this set
        set_accuracy = accuracy_score(set_data["actual_p1_win"], set_data["predicted_p1_win"])
        
        # Average momentum scores
        avg_momentum_p1 = set_data["p1_momentum"].mean()
        avg_momentum_p2 = set_data["p2_momentum"].mean()
        avg_momentum_diff = set_data["momentum_diff"].mean()
        
        # Momentum swings (number of times momentum leader changes)
        momentum_leader = (set_data["momentum_diff"] > 0).astype(int)
        momentum_swings = (momentum_leader.diff() != 0).sum() - 1
        momentum_swings = max(0, momentum_swings)
        
        # Identify games won by each player in this set
        set_data_sorted = set_data.sort_values(["game_no", "point_no"])
        games_won_p1 = set_data_sorted[set_data_sorted["game_victor"] == 1]["game_no"].nunique()
        games_won_p2 = set_data_sorted[set_data_sorted["game_victor"] == 2]["game_no"].nunique()
        
        # Swing prediction metrics
        actual_swings_set = set_data["actual_swing"].sum() if "actual_swing" in set_data.columns else 0
        predicted_swings_set = set_data["predicted_swing"].sum() if "predicted_swing" in set_data.columns else 0
        swing_prediction_accuracy = accuracy_score(
            set_data["actual_swing"].fillna(0),
            set_data["predicted_swing"].fillna(0)
        ) if "actual_swing" in set_data.columns and "predicted_swing" in set_data.columns else None
        
        set_analysis.append({
            "Match_ID": match_id,
            "Set": set_num,
            "Total_Points": total_points,
            "P1_Wins_Actual": p1_wins_actual,
            "P2_Wins_Actual": p2_wins_actual,
            "P1_Wins_Predicted": p1_wins_pred,
            "P2_Wins_Predicted": p2_wins_pred,
            "Model_Accuracy": set_accuracy,
            "Avg_Momentum_P1": avg_momentum_p1,
            "Avg_Momentum_P2": avg_momentum_p2,
            "Avg_Momentum_Diff": avg_momentum_diff,
            "Momentum_Swings": momentum_swings,
            "Actual_Swings": actual_swings_set,
            "Predicted_Swings": predicted_swings_set,
            "Swing_Prediction_Accuracy": swing_prediction_accuracy,
            "Games_Won_P1": games_won_p1,
            "Games_Won_P2": games_won_p2
        })

set_analysis_df = pd.DataFrame(set_analysis)

print("\n--- SET-BY-SET SUMMARY TABLE ---")
print(set_analysis_df.to_string(index=False))

# Save set analysis for report
set_analysis_df.to_csv("set_by_set_analysis.csv", index=False)

# Identify sets with biggest momentum shifts
print("\n--- SETS WITH BIGGEST MOMENTUM SHIFTS ---")
set_analysis_df_sorted = set_analysis_df.sort_values("Momentum_Swings", ascending=False)
print(set_analysis_df_sorted[["Match_ID", "Set", "Momentum_Swings", "Avg_Momentum_Diff"]].head())

# Overall summary by match
print("\n--- OVERALL SUMMARY BY MATCH ---")
for match_id in test_match_ids:
    match_summary = set_analysis_df[set_analysis_df["Match_ID"] == match_id]
    if len(match_summary) == 0:
        continue
    
    match_point_accuracy = accuracy_score(
        test_clean_with_pred[test_clean_with_pred["match_id"] == match_id]["actual_p1_win"],
        test_clean_with_pred[test_clean_with_pred["match_id"] == match_id]["predicted_p1_win"]
    )
    
    print(f"\n{match_id}:")
    print(f"  Total sets: {len(match_summary)}")
    print(f"  Total points: {match_summary['Total_Points'].sum()}")
    print(f"  Point prediction accuracy: {match_point_accuracy:.4f}")
    print(f"  Average momentum swings per set: {match_summary['Momentum_Swings'].mean():.2f}")
    if "Swing_Prediction_Accuracy" in match_summary.columns:
        avg_swing_acc = match_summary["Swing_Prediction_Accuracy"].mean()
        if not pd.isna(avg_swing_acc):
            print(f"  Average swing prediction accuracy: {avg_swing_acc:.4f}")

# Overall summary
print("\n--- OVERALL TEST SET SUMMARY ---")
print(f"Total test matches: {len(test_match_ids)}")
print(f"Total sets: {len(set_analysis_df)}")
print(f"Total points: {set_analysis_df['Total_Points'].sum()}")
print(f"Overall point prediction accuracy: {accuracy:.4f}")
print(f"Overall swing prediction accuracy: {swing_accuracy:.4f}")
print(f"Average momentum swings per set: {set_analysis_df['Momentum_Swings'].mean():.2f}")


# --------------------------------------------------------------
# ADDITIONAL ANALYSIS METRICS
# --------------------------------------------------------------

print("\n--- ADDITIONAL ANALYSIS METRICS ---")

# 1. Momentum correlation with actual point outcomes
momentum_corr_p1 = test_clean_with_pred["p1_momentum"].corr(test_clean_with_pred["actual_p1_win"])
momentum_corr_diff = test_clean_with_pred["momentum_diff"].corr(test_clean_with_pred["actual_p1_win"])
print(f"\n1. Momentum Correlation with Point Outcomes:")
print(f"   P1 Momentum vs P1 Win: {momentum_corr_p1:.4f}")
print(f"   Momentum Diff vs P1 Win: {momentum_corr_diff:.4f}")

# 2. Critical points analysis (break points, set points)
test_clean_with_pred["is_break_point"] = ((test_clean_with_pred["p1_break_pt"] == 1) | 
                                          (test_clean_with_pred["p2_break_pt"] == 1)).astype(int)

# Identify set points (when a player can win the set)
test_clean_with_pred["is_set_point"] = ((test_clean_with_pred["p1_games"] >= 5) | 
                                        (test_clean_with_pred["p2_games"] >= 5)).astype(int)

critical_points = test_clean_with_pred[
    (test_clean_with_pred["is_break_point"] == 1) | 
    (test_clean_with_pred["is_set_point"] == 1)
]

if len(critical_points) > 0:
    critical_accuracy = accuracy_score(
        critical_points["actual_p1_win"], 
        critical_points["predicted_p1_win"]
    )
    print(f"\n2. Critical Points Performance:")
    print(f"   Total critical points: {len(critical_points)}")
    print(f"   Accuracy on critical points: {critical_accuracy:.4f}")
    print(f"   Accuracy on non-critical points: {accuracy_score(test_clean_with_pred[test_clean_with_pred['is_break_point'] == 0]['actual_p1_win'], test_clean_with_pred[test_clean_with_pred['is_break_point'] == 0]['predicted_p1_win']):.4f}")

# 3. Momentum change rate
test_clean_sorted = test_clean_with_pred.sort_values(["set_no", "game_no", "point_no"]).reset_index(drop=True)
momentum_diff_change = test_clean_sorted["momentum_diff"].diff().abs()
avg_momentum_change_rate = momentum_diff_change.mean()
print(f"\n3. Momentum Change Rate:")
print(f"   Average absolute momentum change per point: {avg_momentum_change_rate:.4f}")

# 4. Compare final match momentum patterns vs training matches
train_momentum_stats = {
    "mean_p1": train_features["p1_momentum"].mean(),
    "std_p1": train_features["p1_momentum"].std(),
    "mean_diff": train_features["momentum_diff"].mean(),
    "std_diff": train_features["momentum_diff"].std(),
    "mean_swings": train_features.groupby("match_id").apply(
        lambda x: ((x["momentum_diff"] > 0).astype(int).diff() != 0).sum()
    ).mean()
}

test_momentum_stats = {
    "mean_p1": test_clean_with_pred["p1_momentum"].mean(),
    "std_p1": test_clean_with_pred["p1_momentum"].std(),
    "mean_diff": test_clean_with_pred["momentum_diff"].mean(),
    "std_diff": test_clean_with_pred["momentum_diff"].std(),
    "mean_swings": (test_clean_sorted["momentum_diff"].diff().abs() > 0.1).sum() / len(test_clean_sorted)
}

print(f"\n4. Final Match vs Training Matches Comparison:")
print(f"   P1 Momentum - Training: {train_momentum_stats['mean_p1']:.3f} ± {train_momentum_stats['std_p1']:.3f}")
print(f"   P1 Momentum - Final: {test_momentum_stats['mean_p1']:.3f} ± {test_momentum_stats['std_p1']:.3f}")
print(f"   Momentum Diff - Training: {train_momentum_stats['mean_diff']:.3f} ± {train_momentum_stats['std_diff']:.3f}")
print(f"   Momentum Diff - Final: {test_momentum_stats['mean_diff']:.3f} ± {test_momentum_stats['std_diff']:.3f}")

# Save additional metrics
additional_metrics = pd.DataFrame({
    "Metric": [
        "Momentum Correlation (P1)",
        "Momentum Correlation (Diff)",
        "Critical Points Accuracy",
        "Avg Momentum Change Rate",
        "Training P1 Momentum Mean",
        "Final Match P1 Momentum Mean"
    ],
    "Value": [
        momentum_corr_p1,
        momentum_corr_diff,
        critical_accuracy if len(critical_points) > 0 else None,
        avg_momentum_change_rate,
        train_momentum_stats['mean_p1'],
        test_momentum_stats['mean_p1']
    ]
})
additional_metrics.to_csv("additional_metrics.csv", index=False)


# --------------------------------------------------------------
# ENHANCED VISUALIZATIONS
# --------------------------------------------------------------

print("\n--- CREATING ENHANCED VISUALIZATIONS ---")

# 1. Momentum over time plot with set boundaries
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Momentum scores over time
test_clean_sorted = test_clean_with_pred.sort_values(["set_no", "game_no", "point_no"]).reset_index(drop=True)
point_numbers = range(len(test_clean_sorted))

axes[0].plot(point_numbers, test_clean_sorted["p1_momentum"].values, 
             label='Player 1 Momentum', linewidth=2, alpha=0.8, color='blue')
axes[0].plot(point_numbers, test_clean_sorted["p2_momentum"].values, 
             label='Player 2 Momentum', linewidth=2, alpha=0.8, color='red')
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Mark set boundaries
set_boundaries = []
current_set = None
for idx, set_num in enumerate(test_clean_sorted["set_no"]):
    if current_set is None:
        current_set = set_num
    elif set_num != current_set:
        axes[0].axvline(x=idx, color='black', linestyle=':', alpha=0.5, linewidth=1)
        set_boundaries.append(idx)
        current_set = set_num

axes[0].set_xlabel('Point Number')
axes[0].set_ylabel('Momentum Score')
axes[0].set_title('Momentum Scores Over Final Match (Set Boundaries Marked)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Momentum difference over time
axes[1].plot(point_numbers, test_clean_sorted["momentum_diff"].values, 
              linewidth=2, alpha=0.8, color='green')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[1].fill_between(point_numbers, 0, test_clean_sorted["momentum_diff"].values, 
                      where=(test_clean_sorted["momentum_diff"] > 0), alpha=0.3, color='blue', label='P1 Ahead')
axes[1].fill_between(point_numbers, 0, test_clean_sorted["momentum_diff"].values, 
                      where=(test_clean_sorted["momentum_diff"] < 0), alpha=0.3, color='red', label='P2 Ahead')

# Mark turning points (where momentum crosses zero)
turning_points = []
for idx in range(1, len(test_clean_sorted)):
    if (test_clean_sorted.iloc[idx-1]["momentum_diff"] > 0 and 
        test_clean_sorted.iloc[idx]["momentum_diff"] < 0) or \
       (test_clean_sorted.iloc[idx-1]["momentum_diff"] < 0 and 
        test_clean_sorted.iloc[idx]["momentum_diff"] > 0):
        turning_points.append(idx)
        axes[1].axvline(x=idx, color='orange', linestyle='--', alpha=0.5, linewidth=0.5)

for boundary in set_boundaries:
    axes[1].axvline(x=boundary, color='black', linestyle=':', alpha=0.5, linewidth=1)

axes[1].set_xlabel('Point Number')
axes[1].set_ylabel('Momentum Difference (P1 - P2)')
axes[1].set_title(f'Momentum Difference Over Match (Turning Points: {len(turning_points)})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Set-by-set momentum comparison
set_nums = set_analysis_df["Set"].values
axes[2].bar(set_nums - 0.2, set_analysis_df["Avg_Momentum_P1"].values, 
            width=0.4, label='Avg P1 Momentum', alpha=0.8, color='blue')
axes[2].bar(set_nums + 0.2, set_analysis_df["Avg_Momentum_P2"].values, 
            width=0.4, label='Avg P2 Momentum', alpha=0.8, color='red')
axes[2].set_xlabel('Set Number')
axes[2].set_ylabel('Average Momentum Score')
axes[2].set_title('Average Momentum by Set')
axes[2].set_xticks(set_nums)
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# plt.savefig("momentum_analysis_plots.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature importance visualization
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(15)
if model_name == "Random Forest":
    ax.barh(range(len(top_features)), top_features["importance"].values, color='steelblue')
    ax.set_xlabel('Feature Importance')
else:
    ax.barh(range(len(top_features)), top_features["abs_coefficient"].values, color='steelblue')
    ax.set_xlabel('Absolute Coefficient Value')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features["feature"].values)
ax.set_title(f'Top 15 Most Important Features ({model_name})')
ax.invert_yaxis()
plt.tight_layout()
# plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Prediction accuracy by set
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(set_analysis_df["Set"].values, set_analysis_df["Model_Accuracy"].values, 
       color='green', alpha=0.7, edgecolor='black')
ax.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy ({accuracy:.3f})')
ax.set_xlabel('Set Number')
ax.set_ylabel('Model Accuracy')
ax.set_title('Model Accuracy by Set')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
# plt.savefig("accuracy_by_set.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Momentum swings visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(set_analysis_df["Set"].values, set_analysis_df["Momentum_Swings"].values, 
       color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Set Number')
ax.set_ylabel('Number of Momentum Swings')
ax.set_title('Momentum Swings per Set')
ax.set_xticks(set_analysis_df["Set"].values)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
# plt.savefig("momentum_swings_by_set.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------
# SUMMARY RESULTS FOR REPORT
# --------------------------------------------------------------

print("\n--- SUMMARY RESULTS FOR REPORT ---")

summary_results = {
    "Model_Performance": {
        "Model_Type": model_name,
        "Overall_Accuracy": accuracy,
        "ROC_AUC": auc if 'auc' in locals() else "N/A",
        "Log_Loss": logloss,
        "Baseline_Accuracy": baseline_accuracy,
        "Improvement_Over_Baseline": accuracy - baseline_accuracy
    },
    "Swing_Prediction_Performance": {
        "Swing_Model_Type": swing_model_name if 'swing_model_name' in locals() else "N/A",
        "Swing_Prediction_Accuracy": swing_accuracy if 'swing_accuracy' in locals() else "N/A",
        "Swing_ROC_AUC": swing_auc if 'swing_auc' in locals() else "N/A",
        "Swing_Log_Loss": swing_logloss if 'swing_logloss' in locals() else "N/A",
        "Swing_Rate_Training": float(y_swing_train.mean()) if 'y_swing_train' in locals() else "N/A",
        "Swing_Rate_Test": float(y_swing_test.mean()) if 'y_swing_test' in locals() else "N/A"
    },
    "Momentum_Significance": {
        "P1_Significant_Matches_Pct": momentum_df['p1_momentum_significant'].mean() * 100,
        "P2_Significant_Matches_Pct": momentum_df['p2_momentum_significant'].mean() * 100,
        "P1_Turning_Significant_Pct": momentum_df['p1_turning_significant'].mean() * 100,
        "P2_Turning_Significant_Pct": momentum_df['p2_turning_significant'].mean() * 100
    },
    "Test_Matches_Analysis": {
        "Total_Test_Matches": len(test_match_ids),
        "Test_Match_IDs": test_match_ids,
        "Total_Sets": len(set_analysis_df),
        "Total_Points": int(set_analysis_df['Total_Points'].sum()),
        "Avg_Momentum_Swings_Per_Set": float(set_analysis_df['Momentum_Swings'].mean()),
        "Momentum_Correlation": float(momentum_corr_diff),
        "Critical_Points_Accuracy": float(critical_accuracy) if len(critical_points) > 0 else "N/A"
    },
    "Top_Features_Point_Prediction": feature_importance.head(10)["feature"].tolist(),
    "Top_Features_Swing_Prediction": swing_feature_importance.head(10)["feature"].tolist() if 'swing_feature_importance' in locals() else []
}

import json
# Pretty print the dictionary as a JSON string with 4 spaces indentation
pretty_json_string = json.dumps(summary_results, indent=4)
print(pretty_json_string)



