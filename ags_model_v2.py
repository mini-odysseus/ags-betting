"""
AGS Betting Model v2 - Last 10 Games Form-Based
Simple, equally-weighted last 10 games for both player and team defense
"""

import json
import math
from typing import List, Dict

DATA_PATH = "/workspace/analysis/results.json"
BETTING_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PENALTY_METHODS = ["additive", "distributed"]
TRAIN_YEARS = [15, 16, 17, 18]
VALIDATE_YEARS = [19, 20, 21]

def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    records = []
    for player_record in data:
        player_name = player_record['player']
        for season_data in player_record.get('season_data', []):
            record = {
                'player': player_name,
                'season': season_data['season'],
                'team': season_data['team'],
                'npg': season_data['npg'],
                'npxg': season_data['npxg'],
                'minutes': season_data['minutes'],
                'npxg_per90': season_data.get('npxg_per90', 0),
            }
            records.append(record)
    
    return records

def get_season_year(season_str: str) -> int:
    return int(season_str.split('-')[0])

def calculate_last_10_xg(player_records: List[Dict]) -> float:
    if not player_records:
        return 0.0
    sorted_records = sorted(player_records, key=lambda x: x['season'])
    last_10 = sorted_records[-10:]
    xg_per_90s = [r['npxg_per90'] for r in last_10 if r.get('npxg_per90')]
    if not xg_per_90s:
        return 0.0
    return sum(xg_per_90s) / len(xg_per_90s)

def calculate_team_last_10_defense(team: str, all_records: List[Dict]) -> float:
    team_records = [r for r in all_records if r['team'] == team]
    if not team_records:
        return 1.0
    sorted_records = sorted(team_records, key=lambda x: x['season'])
    last_10 = sorted_records[-10:]
    xg_conceded = [r['npxg'] for r in last_10]
    avg_conceded = sum(xg_conceded) / len(xg_conceded) if xg_conceded else 1.0
    all_xg = [r['npxg'] for r in all_records]
    league_avg = sum(all_xg) / len(all_xg) if all_xg else 1.0
    return avg_conceded / league_avg if league_avg > 0 else 1.0

def predict_ags(player: str, history: List[Dict], opponent_defense: float, penalty_method: str):
    base_xg_90 = calculate_last_10_xg(history)
    adjusted_xg_90 = base_xg_90 * opponent_defense
    adjusted_xg = adjusted_xg_90 * (80 / 90)
    
    if penalty_method == "additive":
        adjusted_xg += 0.05
    else:
        adjusted_xg += 0.005
    
    prob = 1 - math.exp(-adjusted_xg)
    return {'prob': prob, 'xg': adjusted_xg}

def brier_score(preds, outcomes):
    if not preds:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / len(preds)

def log_loss(preds, outcomes):
    if not preds:
        return 0.0
    eps = 1e-15
    losses = []
    for p, o in zip(preds, outcomes):
        p_clip = max(eps, min(1 - eps, p))
        losses.append(-(o * math.log(p_clip) + (1 - o) * math.log(1 - p_clip)))
    return sum(losses) / len(losses)

def main():
    print("=" * 60)
    print("AGS MODEL v2 - Last 10 Games Form")
    print("=" * 60)
    
    records = load_data(DATA_PATH)
    print(f"Loaded {len(records)} records")
    
    train = [r for r in records if get_season_year(r['season']) in TRAIN_YEARS]
    validate = [r for r in records if get_season_year(r['season']) in VALIDATE_YEARS]
    
    print(f"Train: {len(train)}, Validate: {len(validate)}")
    
    for method in PENALTY_METHODS:
        print(f"\n{method.upper()}:")
        
        player_hist = {}
        for r in train:
            p = r['player']
            if p not in player_hist:
                player_hist[p] = []
            player_hist[p].append(r)
        
        preds = []
        outcomes = []
        
        for r in validate:
            defense = calculate_team_last_10_defense(r['team'], train)
            history = player_hist.get(r['player'], [])
            pred = predict_ags(r['player'], history, defense, method)
            preds.append(pred['prob'])
            outcomes.append(1 if r['npg'] > 0 else 0)
        
        brier = brier_score(preds, outcomes)
        ll = log_loss(preds, outcomes)
        
        baseline_prob = sum(outcomes) / len(outcomes) if outcomes else 0.3
        baseline_preds = [baseline_prob] * len(outcomes)
        baseline_brier = brier_score(baseline_preds, outcomes)
        bss = 1 - (brier / baseline_brier) if baseline_brier > 0 else 0
        
        print(f"  Brier: {brier:.4f}, LogLoss: {ll:.4f}, BSS: {bss:.4f}")

if __name__ == "__main__":
    main()
