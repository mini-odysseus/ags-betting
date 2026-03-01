from data_loader import load_and_structure_data
from feature_engineering import calculate_xg_per_90, calculate_opponent_defense_rating, weight_by_expected_minutes, handle_penalties_additive, handle_penalties_distributed
from model import poisson_model, negative_binomial_model, ags_probability, brier_skill_score, train_validate
from visualization import plot_reliability_diagram, plot_threshold_analysis, create_calibration_curve
import numpy as np
import pandas as pd


def main():
    # Load data
    data = load_and_structure_data('/workspace/analysis/results.json')

    # Ensure the data includes all needed fields for training
    required_fields = ['player', 'season', 'team', 'minutes', 'npg', 'npxg']
    if not all(field in data.columns for field in required_fields):
        print(f"Error: Data does not contain all required fields: {required_fields}")
        return

    # Define training and validation years
    train_years = list(range(2014, 2019))
    validate_years = list(range(2019, 2023))

    #Betting thresholds to test
    betting_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    penalty_methods = ["additive", "distributed"]
    results = {}

    for method in penalty_methods:
        print(f"\nRunning with {method.capitalize()} Penalty Handling...")
        brier_score, log_loss_value, skill_score, baseline_brier, baseline_log_loss = train_validate(
            data.to_dict('records'),  # train_validate expects list of dicts
            train_years, validate_years, method
        )
        results[method] = {
            "brier_score": brier_score,
            "log_loss": log_loss_value,
            "skill_score": skill_score,
            "baseline_brier": baseline_brier,
            "baseline_logloss": baseline_log_loss
        }
        print(f"{method.capitalize()} Penalty - Brier Score: {brier_score}, Log Loss: {log_loss_value}, Skill Score: {skill_score}")
        print(f"Baseline Brier: {baseline_brier}, Baseline LogLoss: {baseline_log_loss}")
      
        if 'season_data' in data.columns:
            validation_data = [d for d in data.to_dict('records') if int(d["season"][:4]) in validate_years]

            #Extract probabilites and outcomes for the reliability diagram
            probabilites = [ags_probability(d['npxg']) for d in validation_data ]
            outcomes = [1 if d['npg']>0 else 0 for d in validation_data]

            #Plot Reliability Diagram
            prob_bins, observed_bins = create_calibration_curve(probabilites, outcomes, n_bins=10)
            plot_reliability_diagram(prob_bins, observed_bins, output_file=f"{method}_reliability_diagram.png")

            #Placeholder Threshold analysis. Needs more data
            threshold_metric_values = np.random.rand(len(betting_thresholds))
            plot_threshold_analysis(betting_thresholds, threshold_metric_values,  output_file=f"{method}_threshold_analysis.png")


    # Compare the two methods
    if results["additive"]["skill_score"] > results["distributed"]["skill_score"]:
        print("\nAdditive penalty handling performs better.")
    else:
        print("\nDistributed penalty handling performs better.")


if __name__ == "__main__":
    main()
