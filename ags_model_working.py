"""
AGS Betting Model - Working Implementation
Uses only standard library to avoid dependency issues
"""

import json
import math
import os
from typing import List, Dict, Tuple

# Configuration
DATA_PATH = "/workspace/analysis/results.json"
OUTPUT_DIR = "/workspace/ags-betting/output"
TRAIN_YEARS = [15, 16, 17, 18]  # 15-16 through 18-19
VALIDATE_YEARS = [19, 20, 21]   # 19-20 through 21-22
BETTING_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PENALTY_METHODS = ["additive", "distributed"]


def load_data(file_path: str) -> List[Dict]:
    """Load player-season data from results.json"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Flatten to player-season records
    records = []
    for player_record in data:
        player_name = player_record['player']
        for season_data in player_record.get('season_data', []):
            record = {
                'player': player_name,
                'season': season_data['season'],  # Format: '15-16'
                'team': season_data['team'],
                'npg': season_data['npg'],  # Non-penalty goals
                'npxg': season_data['npxg'],  # Non-penalty xG
                'minutes': season_data['minutes'],
                'npg_per90': season_data.get('npg_per90', 0),
                'npxg_per90': season_data.get('npxg_per90', 0),
                'shots': season_data.get('shots', 0)
            }
            records.append(record)
    
    return records


def get_season_year(season_str: str) -> int:
    """Extract start year from season string '15-16' -> 15"""
    return int(season_str.split('-')[0])


def calculate_ema(values: List[float], span: int = 10) -> float:
    """Calculate Exponential Moving Average"""
    if not values:
        return 0.0
    alpha = 2 / (span + 1)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema


def calculate_xg_per_90(player_history: List[Dict]) -> float:
    """Calculate EMA-weighted xG per 90 for a player"""
    if not player_history:
        return 0.0
    
    # Sort by season
    sorted_history = sorted(player_history, key=lambda x: x['season'])
    xg_per_90s = [h['npxg_per90'] for h in sorted_history if h.get('npxg_per90')]
    
    if not xg_per_90s:
        return 0.0
    
    return calculate_ema(xg_per_90s, span=10)


def calculate_team_defense_ratings(records: List[Dict]) -> Dict[str, float]:
    """Calculate team defensive ratings based on xG conceded"""
    team_stats = {}
    
    for record in records:
        team = record['team']
        season = record['season']
        key = f"{team}_{season}"
        
        if key not in team_stats:
            team_stats[key] = {'total_npxg': 0, 'count': 0}
        
        team_stats[key]['total_npxg'] += record['npxg']
        team_stats[key]['count'] += 1
    
    # Calculate league average
    league_avg = sum(s['total_npxg'] / s['count'] for s in team_stats.values()) / len(team_stats) if team_stats else 1.0
    
    # Defense rating: higher = weaker defense
    defense_ratings = {}
    for key, stats in team_stats.items():
        avg_xg = stats['total_npxg'] / stats['count'] if stats['count'] > 0 else 0
        defense_ratings[key] = avg_xg / league_avg if league_avg > 0 else 1.0
    
    return defense_ratings


def predict_ags_probability(
    player: str,
    player_history: List[Dict],
    opponent_defense: float,
    expected_minutes: float = 80,
    penalty_xg: float = 0.05,
    penalty_method: str = "additive"
) -> Dict:
    """Predict Anytime Goalscorer probability for a player"""
    
    # Get base xG/90 from historical EMA
    base_xg_90 = calculate_xg_per_90(player_history)
    
    # Adjust for opponent defense
    adjusted_xg_90 = base_xg_90 * opponent_defense
    
    # Scale for expected minutes
    adjusted_xg = adjusted_xg_90 * (expected_minutes / 90)
    
    # Handle penalties
    if penalty_method == "additive":
        adjusted_xg += penalty_xg
    else:
        adjusted_xg += penalty_xg * 0.1  # Distributed
    
    # Calculate P(at least 1 goal) = 1 - exp(-lambda)
    prob_anytime = 1 - math.exp(-adjusted_xg)
    
    return {
        'player': player,
        'base_xg_90': base_xg_90,
        'adjusted_xg': adjusted_xg,
        'prob_anytime': prob_anytime,
        'expected_minutes': expected_minutes,
        'opponent_defense': opponent_defense,
        'penalty_method': penalty_method
    }


def calculate_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate mean squared error between predictions and outcomes"""
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        return 0.0
    
    squared_errors = [(p - o) ** 2 for p, o in zip(predictions, outcomes)]
    return sum(squared_errors) / len(squared_errors)


def calculate_log_loss(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate log loss"""
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        return 0.0
    
    epsilon = 1e-15
    losses = []
    for p, o in zip(predictions, outcomes):
        p_clipped = max(epsilon, min(1 - epsilon, p))
        loss = -(o * math.log(p_clipped) + (1 - o) * math.log(1 - p_clipped))
        losses.append(loss)
    
    return sum(losses) / len(losses)


def calculate_brier_skill_score(brier_score: float, baseline_brier: float) -> float:
    """Calculate Brier Skill Score vs naive baseline"""
    if baseline_brier == 0:
        return 0.0
    return 1 - (brier_score / baseline_brier)


def train_validate(records, train_years, validate_years, penalty_method):
    """Time-series cross-validation"""
    
    # Split data by season year
    train_data = [r for r in records if get_season_year(r['season']) in train_years]
    validate_data = [r for r in records if get_season_year(r['season']) in validate_years]
    
    print(f"  Training on {len(train_data)} player-seasons")
    print(f"  Validating on {len(validate_data)} player-seasons")
    
    if not validate_data:
        return 0.0, 0.0, 0.0, 0.25, 0.693
    
    # Calculate team defense ratings from training data
    defense_ratings = calculate_team_defense_ratings(train_data)
    
    # Group training data by player for history lookup
    player_histories = {}
    for record in train_data:
        player = record['player']
        if player not in player_histories:
            player_histories[player] = []
        player_histories[player].append(record)
    
    # Make predictions on validation set
    predictions = []
    outcomes = []
    
    # Calculate league average for naive baseline
    league_avg_xg_90 = sum(r['npxg_per90'] for r in train_data if r.get('npxg_per90')) / len(train_data) if train_data else 0.3
    naive_prob = 1 - math.exp(-league_avg_xg_90 * (80/90))
    
    for record in validate_data:
        player = record['player']
        team_season = f"{record['team']}_{record['season']}"
        
        opponent_defense = defense_ratings.get(team_season, 1.0)
        history = player_histories.get(player, [])
        
        pred = predict_ags_probability(
            player=player,
            player_history=history,
            opponent_defense=opponent_defense,
            expected_minutes=80,
            penalty_xg=0.05,
            penalty_method=penalty_method
        )
        
        predictions.append(pred['prob_anytime'])
        outcomes.append(1 if record['npg'] > 0 else 0)
    
    # Calculate metrics
    brier = calculate_brier_score(predictions, outcomes)
    logloss = calculate_log_loss(predictions, outcomes)
    
    # Naive baseline predictions
    baseline_predictions = [naive_prob] * len(outcomes)
    baseline_brier = calculate_brier_score(baseline_predictions, outcomes)
    baseline_logloss = calculate_log_loss(baseline_predictions, outcomes)
    
    # Skill score
    skill = calculate_brier_skill_score(brier, baseline_brier)
    
    return brier, logloss, skill, baseline_brier, baseline_logloss, predictions, outcomes


def test_betting_thresholds(predictions, outcomes, thresholds):
    """Test multiple betting value thresholds"""
    results = []
    
    # Synthetic market probabilities
    market_probs = [max(0.01, min(0.99, p * 0.9 + 0.05)) for p in predictions]
    
    for threshold in thresholds:
        value_edges = [(p / m) - 1 for p, m in zip(predictions, market_probs)]
        bets = [(i, p, o) for i, (p, o, edge) in enumerate(zip(predictions, outcomes, value_edges)) if edge >= threshold]
        
        if not bets:
            results.append({
                'threshold': threshold,
                'n_bets': 0,
                'hit_rate': 0.0,
                'avg_edge': 0.0,
                'expected_roi': -1.0
            })
            continue
        
        bet_outcomes = [b[2] for b in bets]
        hit_rate = sum(bet_outcomes) / len(bet_outcomes)
        avg_edge = sum(value_edges[b[0]] for b in bets) / len(bets)
        
        avg_odds = sum(1/market_probs[b[0]] for b in bets) / len(bets)
        expected_roi = (hit_rate * avg_odds) - 1
        
        results.append({
            'threshold': threshold,
            'n_bets': len(bets),
            'hit_rate': hit_rate,
            'avg_edge': avg_edge,
            'expected_roi': expected_roi
        })
    
    return results


def main():
    print("=" * 70)
    print("AGS BETTING MODEL - Complete Implementation")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    records = load_data(DATA_PATH)
    print(f"   Loaded {len(records)} player-season records")
    
    # Run for both penalty methods
    results_summary = {}
    
    for method in PENALTY_METHODS:
        print(f"\n{'='*70}")
        print(f"PENALTY METHOD: {method.upper()}")
        print(f"{'='*70}")
        
        print("\n2. Running time-series cross-validation...")
        brier, logloss, skill, baseline_brier, baseline_logloss, predictions, outcomes = train_validate(
            records, TRAIN_YEARS, VALIDATE_YEARS, method
        )
        
        print(f"\n   Validation Results:")
        print(f"     Brier Score: {brier:.4f}")
        print(f"     Log Loss: {logloss:.4f}")
        print(f"     Brier Skill Score: {skill:.4f}")
        print(f"     Baseline Brier: {baseline_brier:.4f}")
        print(f"     Baseline Log Loss: {baseline_logloss:.4f}")
        
        results_summary[method] = {
            'brier': brier,
            'logloss': logloss,
            'skill': skill,
            'baseline_brier': baseline_brier,
            'baseline_logloss': baseline_logloss
        }
        
        # Test betting thresholds
        print("\n3. Testing betting thresholds...")
        threshold_results = test_betting_thresholds(predictions, outcomes, BETTING_THRESHOLDS)
        
        print("\n   Threshold Analysis:")
        print(f"   {'Threshold':<10} {'N Bets':<10} {'Hit Rate':<12} {'Avg Edge':<12} {'Exp ROI':<10}")
        print("   " + "-" * 60)
        for r in threshold_results:
            print(f"   {r['threshold']:<10.2f} {r['n_bets']:<10} {r['hit_rate']:<12.4f} {r['avg_edge']:<12.4f} {r['expected_roi']:<10.4f}")
    
    # Compare methods
    print(f"\n{'='*70}")
    print("METHOD COMPARISON")
    print(f"{'='*70}")
    
    for method in PENALTY_METHODS:
        r = results_summary[method]
        print(f"\n{method.upper()}:")
        print(f"  Brier Skill Score: {r['skill']:.4f}")
        print(f"  Log Loss: {r['logloss']:.4f}")
    
    better_method = "Additive" if results_summary["additive"]["skill"] > results_summary["distributed"]["skill"] else "Distributed"
    print(f"\nBetter Method: {better_method}")
    print(f"  (Additive BSS: {results_summary['additive']['skill']:.4f}, Distributed BSS: {results_summary['distributed']['skill']:.4f})")


if __name__ == "__main__":
    main()
