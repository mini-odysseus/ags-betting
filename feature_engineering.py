
import numpy as np
import pandas as pd

def calculate_xg_per_90(player_xgs, minutes_played):
    # Calculate xG/90 with EMA weighting (span=10)
    player_xgs = pd.Series(player_xgs)
    minutes_played = pd.Series(minutes_played)
    combined = pd.DataFrame({'xG': player_xgs, 'minutes': minutes_played})
    combined = combined.sort_index(ascending=False) # reverse chronological order
    xg_per_90 = (combined['xG'] / (combined['minutes'] / 90)).fillna(0) # handle zero minutes
    ema = xg_per_90.ewm(span=10, adjust=False).mean()  # Exponential Moving Average
    ema = ema.sort_index(ascending=True) # restore chronological order 
    return ema

def calculate_opponent_defense_rating(team_conceded_xgs):
    # Calculate opponent defense rating based on conceded xG
     team_conceded_xgs = pd.Series(team_conceded_xgs)
     mean_conceded_xg = team_conceded_xgs.mean()
     if mean_conceded_xg == 0:
       return pd.Series([1.0]*len(team_conceded_xgs)) # avoid division by zero. Assume neutral if no data
     return team_conceded_xgs / mean_conceded_xg

def weight_by_expected_minutes(xg_per_90, expected_minutes):
    # Weight xG/90 by expected minutes
    return xg_per_90 * (expected_minutes / 90)

def handle_penalties_additive(player_xg, penalty_xg):
    # Add penalty xG directly to player's xG
    return player_xg + penalty_xg

def handle_penalties_distributed(team_starters_xgs, player_xg_no_penalty, penalty_xg):
    # Distribute penalty xG across team starters based on their xG contribution
    team_starters_xgs = pd.Series(team_starters_xgs)
    total_xg = team_starters_xgs.sum()
    if total_xg == 0:
        weights = pd.Series([1/len(team_starters_xgs)] * len(team_starters_xgs)) # Distribute evenly if total xG is zero
    else:
        weights = team_starters_xgs / total_xg
    
    distributed_xg = player_xg_no_penalty + weights * penalty_xg
    return distributed_xg

if __name__ == '__main__':
    # Example usage
    player_xgs = [0.5, 0.6, 0.7, 0.8, 0.9]
    minutes_played = [60, 70, 80, 90, 90]
    opponent_defensive_strengths = [0.8, 0.9, 0.7, 0.8, 0.9]
    expected_minutes = 75
    penalty_xg = 0.2
    team_starters_xgs = [0.4, 0.5, 0.6, 0.7, 0.8]
    player_xg_no_penalty = 0.5
    team_conceded_xgs = [1.0, 1.2, 0.9, 1.1, 1.3]

    xg_per_90 = calculate_xg_per_90(player_xgs, minutes_played)
    print(f"xG/90: {xg_per_90}")

    opponent_defense_ratings = calculate_opponent_defense_rating(team_conceded_xgs)
    print(f"Opponent Defense Ratings: {opponent_defense_ratings}")

    weighted_xg = weight_by_expected_minutes(xg_per_90, expected_minutes)
    print(f"Weighted xG/90: {weighted_xg}")

    additive_xg = handle_penalties_additive(player_xg_no_penalty, penalty_xg)
    print(f"Additive penalty xG: {additive_xg}")

    distributed_xg = handle_penalties_distributed(team_starters_xgs, player_xg_no_penalty, penalty_xg)
    print(f"Distributed penalty xG: {distributed_xg}")
