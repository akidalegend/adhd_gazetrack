
import pandas as pd
import numpy as np
from analysis import load_csv, _select_gaze_series, extract_stimuli_from_csv
from gaze_tracking.saccades import detect_saccades

csv_path = "sessions/raw/TEST02_prosaccade_20251125_162331.csv"
df = load_csv(csv_path)
stimuli_times = extract_stimuli_from_csv(df)

print(f"Found {len(stimuli_times)} stimuli.")

# Use the same parameters as the last run
vel_thresh = 200.0 
min_dur = 0.015
smooth_w = 5

times, pos, adjusted_thresh, is_calibrated = _select_gaze_series(df, "TEST02", vel_thresh)
print(f"Using threshold: {adjusted_thresh}")

saccades = detect_saccades(times, pos, vel_thresh=adjusted_thresh, min_dur=min_dur, smooth_w=smooth_w)
print(f"Detected {len(saccades)} saccades.")

print("\n--- Detailed Alignment ---")
for i, stim_t in enumerate(stimuli_times):
    print(f"\nTrial {i}: Stimulus at {stim_t:.3f}s")
    
    # Find saccades near this stimulus
    near_saccades = [s for s in saccades if abs(s['onset_t'] - stim_t) < 2.0]
    
    if not near_saccades:
        print("  No saccades detected within +/- 2.0s")
    else:
        for s in near_saccades:
            onset = s['onset_t']
            offset = s['offset_t']
            dur = s['duration']
            peak = s['peak_velocity']
            latency = onset - stim_t
            
            match_str = "MATCH" if (0 <= latency <= 1.0) else "NO MATCH"
            print(f"  Saccade: onset={onset:.3f}s (lat={latency:.3f}s), dur={dur:.3f}s, peak_vel={peak:.1f} -> {match_str}")
            if latency < 0:
                print("    (Anticipatory / Early)")
            elif latency > 1.0:
                print("    (Too late)")

