# Last 100 Shots Model v3
import json, math
from typing import List, Dict

DATA_PATH = "/workspace/analysis/results.json"
PENALTY_METHODS = ["additive", "distributed"]

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    records = []
    for p in data:
        for s in p.get('season_data', []):
            records.append({
                'player': p['player'],
                'season': s['season'],
                'team': s['team'],
                'npg': s['npg'],
                'npxg': s['npxg'],
                'shots': s.get('shots', 0),
                'npxg_per90': s.get('npxg_per90', 0)
            })
    return records

def get_year(s): return int(s.split('-')[0])

def calc_conversion(records):
    if not records: return 0.0
    sorted_r = sorted(records, key=lambda x: x['season'])
    shots, xg = 0, 0
    for r in reversed(sorted_r):
        shots += r.get('shots', 0)
        xg += r.get('npxg', 0)
        if shots >= 100: break
    return xg / shots if shots > 0 else 0.0

def calc_defense(team, all_rec):
    team_rec = [r for r in all_rec if r['team'] == team]
    if not team_rec: return 1.0
    sorted_r = sorted(team_rec, key=lambda x: x['season'])
    shots, xg = 0, 0
    for r in reversed(sorted_r):
        shots += r.get('shots', 0)
        xg += r.get('npxg', 0)
        if shots >= 100: break
    if shots == 0: return 1.0
    avg_xg = xg / shots
    all_shots = sum(r.get('shots', 0) for r in all_rec)
    all_xg = sum(r.get('npxg', 0) for r in all_rec)
    league_avg = all_xg / all_shots if all_shots else 0.1
    return avg_xg / league_avg if league_avg else 1.0

def predict(player, history, defense, method):
    conv = calc_conversion(history)
    xg = conv * 2.5 * defense
    if method == "additive": xg += 0.05
    else: xg += 0.005
    return 1 - math.exp(-xg)

def brier(preds, outs):
    if not preds: return 0.0
    return sum((p-o)**2 for p,o in zip(preds, outs)) / len(preds)

def main():
    rec = load_data(DATA_PATH)
    train = [r for r in rec if get_year(r['season']) in [15,16,17,18]]
    val = [r for r in rec if get_year(r['season']) in [19,20,21]]
    print(f"Last 100 Shots v3 - Train: {len(train)}, Val: {len(val)}")
    
    for method in PENALTY_METHODS:
        print(f"\n{method.upper()}:")
        ph = {}
        for r in train:
            p = r['player']
            if p not in ph: ph[p] = []
            ph[p].append(r)
        preds, outs = [], []
        for r in val:
            d = calc_defense(r['team'], train)
            h = ph.get(r['player'], [])
            prob = predict(r['player'], h, d, method)
            preds.append(prob)
            outs.append(1 if r['npg'] > 0 else 0)
        b = brier(preds, outs)
        base = sum(outs) / len(outs) if outs else 0.3
        base_brier = sum((base-o)**2 for o in outs) / len(outs) if outs else 0.25
        bss = 1 - (b / base_brier) if base_brier else 0
        print(f"  Brier: {b:.4f}, BSS: {bss:.4f}")

if __name__ == "__main__":
    main()
