import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis import load_calibration_model, apply_calibration

def load_data(csv_path, summary_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load Summary/Stimuli
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Extract Stimuli Times
    stimuli = []
    if 'stimuli_directions' in summary:
        stimuli = summary['stimuli_directions']
    
    return df, stimuli, summary.get('session_label', 'Unknown')

def plot_session(csv_path, summary_path):
    df, stimuli, label = load_data(csv_path, summary_path)
    
    # Apply Calibration if available
    model = load_calibration_model(label)
    is_calibrated = apply_calibration(df, model)
    
    # Select Signal
    if is_calibrated:
        signal = df['gaze_x_px']
        ylabel = "Gaze Position (Pixels)"
        title_suffix = "(Calibrated)"
    else:
        signal = df['g_horizontal']
        ylabel = "Gaze Ratio"
        title_suffix = "(Raw)"

    time = df['t']
    
    plt.figure(figsize=(14, 6))
    
    # 1. Plot the Gaze Signal
    plt.plot(time, signal, label='Gaze X', color='blue', alpha=0.7, linewidth=1)
    
    # 2. Plot Stimulus Onsets
    for i, stim in enumerate(stimuli):
        onset = stim.get('time', 0)
        target_x = stim.get('x', 0)
        
        # Draw vertical line for onset
        plt.axvline(x=onset, color='green', linestyle='--', alpha=0.5)
        
        # Label the trial
        y_pos = signal.mean()
        plt.text(onset, y_pos, f"T{i}", rotation=90, verticalalignment='bottom', fontsize=8)
        
        # Optional: Plot the target position as a horizontal segment (if calibrated)
        if is_calibrated:
            # Draw a short horizontal line representing where they SHOULD look
            plt.hlines(y=target_x, xmin=onset, xmax=onset+1.5, color='red', linestyle=':', alpha=0.5)

    # 3. Highlight Blinks
    if 'is_blinking' in df.columns:
        blinks = df[df['is_blinking'] == 1]
        plt.scatter(blinks['t'], [signal.min()] * len(blinks), color='red', marker='x', s=10, label='Blink')

    plt.title(f"Session Visualization: {label} {title_suffix}")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to raw CSV file")
    parser.add_argument("summary_path", help="Path to summary JSON file")
    args = parser.parse_args()
    
    plot_session(args.csv_path, args.summary_path)