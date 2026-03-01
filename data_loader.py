
import json
import pandas as pd

def load_and_structure_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    structured_data = []
    for player_data in data:
        player_name = player_data['player']
        for season_data in player_data['season_data']:
            season = season_data['season']
            team = season_data['team']
            minutes = season_data['minutes']
            npg = season_data['npg']
            npxg = season_data['npxg']

            structured_data.append({
                'player': player_name,
                'season': season,
                'team': team,
                'minutes': minutes,
                'npg': npg,
                'npxg': npxg
            })

    df = pd.DataFrame(structured_data)
    return df

if __name__ == '__main__':
    df = load_and_structure_data('/workspace/analysis/results.json')
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(df.head())
