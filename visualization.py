import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_reliability_diagram(predicted_probabilities, observed_frequencies, output_file = "reliability_diagram.png"):
    # Plot a reliability diagram
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_probabilities, observed_frequencies, marker='o', linestyle='-')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_threshold_analysis(thresholds, metric_values, output_file = "threshold_analysis.png"):
    # Plot a graph of metric values vs. betting thresholds
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, metric_values, marker='o', linestyle='-')
    plt.xlabel("Betting Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Analysis")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def create_calibration_curve(probabilities, outcomes, n_bins=10):
    #Bin the probabilities into n_bins and calculate the mean predicted probability and observed frequency for each bin.
    df = pd.DataFrame({"probability": probabilities, "outcome": outcomes})
    df["bin"] = pd.cut(df["probability"], bins=n_bins, labels=False)
    grouped = df.groupby("bin").agg({"probability": "mean", "outcome": "mean"})
    return grouped["probability"].values, grouped["outcome"].values

if __name__ == '__main__':
    # Example usage
    predicted_probabilities = np.linspace(0, 1, 100)
    observed_frequencies = predicted_probabilities + np.random.normal(0, 0.1, 100)
    observed_frequencies = np.clip(observed_frequencies, 0, 1)

    #Create a calibration curve from raw data
    prob_bins, observed_bins = create_calibration_curve(predicted_probabilities, observed_frequencies, n_bins=10)

    thresholds = np.linspace(0.05, 0.30, 6)
    metric_values = np.random.rand(6)

    plot_reliability_diagram(prob_bins, observed_bins)
    plot_threshold_analysis(thresholds, metric_values)
