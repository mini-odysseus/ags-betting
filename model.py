import numpy as np
from scipy.stats import poisson, nbinom
from sklearn.metrics import brier_score_loss, log_loss
import math

def poisson_model(team_xg, goals_scored):
    # Calculate Poisson probability for goals scored given expected goals
    return poisson.pmf(goals_scored, team_xg)

def negative_binomial_model(team_xg, goals_scored, dispersion=1.0):
    # Calculate Negative Binomial probability for goals scored given expected goals and dispersion
    # Adjust the parameters for the negative binomial distribution
    mu = team_xg
    theta = mu / dispersion
    r = theta
    p = r / (r + mu)
    return nbinom.pmf(goals_scored, n=r, p=p)

def ags_probability(adjusted_xg):
    # Calculate the probability of a player scoring at least one goal
    return 1 - np.exp(-adjusted_xg)

def ags_probability_safe(adjusted_xg):
    # Calculate the probability of a player scoring at least one goal, handling overflow
    if adjusted_xg > 700: # avoid overflow on the exp function
      return 1.0
    return 1 - np.exp(-adjusted_xg)

def brier_skill_score(brier_score, baseline_brier_score):
    # Calculate Brier Skill Score relative to a baseline
    return 1 - (brier_score / baseline_brier_score)

def train_validate(data, train_years, validate_years, penalty_method, naive_baseline = 0.20, verbose = False):
    # Time-series cross-validation
    train_data = [d for d in data if int(d["season"][:4]) in train_years]
    validate_data = [d for d in data if int(d["season"][:4]) in validate_years]

    if verbose:
      print(f"Training on {len(train_data)} matches from {train_years}")
      print(f"Validating on {len(validate_data)} matches from {validate_years}")

    # Train Naive Baseline (average scoring rate on train years)
    player_scoring_rates = [d['npg'] / (d['minutes']/90) for d in train_data if d['minutes'] > 0 ]
    if not player_scoring_rates:
      baseline_brier_score = 0.25  # Example value if no data
      baseline_log_loss = 0.69 # log loss for 50% prob
    else:
      avg_scoring_rate = np.mean(player_scoring_rates) 
      baseline_predictions = [ min(1.0, avg_scoring_rate) for _ in validate_data] # probability never exceeds 1
      actual_values = [1 if d['npg'] > 0 else 0 for d in validate_data] # Did they score at least 1 goal?
      baseline_brier_score = brier_score_loss(actual_values, baseline_predictions)
      baseline_log_loss = log_loss(actual_values, baseline_predictions)


    # Train and Validate model
    predictions = []
    actual_values = []
    for match in validate_data:
      # Implement your AGS probability calculation here using adjusted_xg
      # Placeholder: Replace with your actual model prediction
      adjusted_xg = match['npxg']  # for testing
      predicted_prob = ags_probability_safe(adjusted_xg) # Probability of scoring at least 1 goal

      predictions.append(predicted_prob)
      actual_values.append(1 if match['npg'] > 0 else 0) # Did they score at least 1 goal?

    brier_score = brier_score_loss(actual_values, predictions)
    log_loss_value = log_loss(actual_values, predictions)

    # Calculate Brier Skill Score
    skill_score = brier_skill_score(brier_score, baseline_brier_score)

    return brier_score, log_loss_value, skill_score, baseline_brier_score, baseline_log_loss

if __name__ == '__main__':
    # Example usage
    team_xg = 1.5
    goals_scored = 2
    adjusted_xg = 0.3

    poisson_prob = poisson_model(team_xg, goals_scored)
    print(f"Poisson Probability: {poisson_prob}")

    neg_binomial_prob = negative_binomial_model(team_xg, goals_scored)
    print(f"Negative Binomial Probability: {neg_binomial_prob}")

    ags_prob = ags_probability(adjusted_xg)
    print(f"Anytime Goalscorer Probability: {ags_prob}")

    #Example validation run
    import data_loader
    data = data_loader.load_and_structure_data('/workspace/analysis/results.json')
    train_years = range(2014, 2019)
    validate_years = [2019]
    brier, logloss, skill, base_brier, base_logloss = train_validate(data, train_years, validate_years, "additive", verbose = False)
    print(f"Brier score: {brier}, LogLoss: {logloss}, Skill Score: {skill}")
    print(f"Baseline Brier: {base_brier}, Baseline LogLoss: {base_logloss}")
