import math
import json
import csv
import os
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        'pandas is required for analysis.py. Install it via "pip install pandas" inside your active environment.'
    ) from exc
import numpy as np

from gaze_tracking.saccades import (
    detect_saccades,
    detect_fixations,
    saccade_latency_to_stimuli,
    count_intrusive_saccades,
)

def load_calibration_model(label, calibration_dir="sessions/calibration"):
    """
    Tries to load a calibration JSON for the given label.
    Returns the model dict or None.
    """
    filename = os.path.join(calibration_dir, f"{label}_calibration.json")
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return data.get("model")
        except Exception as e:
            print(f"Warning: Failed to load calibration file {filename}: {e}")
    return None

def apply_calibration(df, model):
    """
    Applies linear calibration model to gaze data.
    Adds 'gaze_x_px' column to dataframe.
    """
    if model and 'g_horizontal' in df.columns:
        # x_screen = slope * gaze_h + intercept
        slope = model.get("x_slope", 1.0)
        intercept = model.get("x_intercept", 0.0)
        
        # Apply model
        df['gaze_x_px'] = df['g_horizontal'] * slope + intercept
        return True
    return False

def load_csv(path):
    df = pd.read_csv(path)
    # ensure sorted by time
    df = df.sort_values('t').reset_index(drop=True)
    # fill missing values to avoid breaks in continuous signals
    df = df.ffill().fillna(0)
    return df

def angle_path_length(df):
    cols = ['yaw', 'pitch', 'roll']
    if not all(c in df.columns for c in cols):
        return 0.0, np.array([])
    a = df[cols].to_numpy()
    diffs = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return float(np.nansum(diffs)), diffs

def angular_speed_stats(diffs, times):
    if len(diffs) == 0:
        return {'mean_speed': 0.0, 'speeds': np.array([])}
    dt = np.diff(times)
    dt[dt == 0] = 1e-6
    speeds = diffs / dt
    return {
        'mean_speed': float(np.nanmean(speeds)),
        'median_speed': float(np.nanmedian(speeds)),
        'max_speed': float(np.nanmax(speeds)),
        'speeds': speeds
    }

def compute_features(csv_path, spike_threshold=30.0, motion_threshold=5.0, df=None):
    df = load_csv(csv_path) if df is None else df
    duration = df['t'].iloc[-1] - df['t'].iloc[0] if len(df) > 1 else 0.0

    path_len, diffs = angle_path_length(df)
    times = df['t'].to_numpy()
    speed_stats = angular_speed_stats(diffs, times)

    speeds = speed_stats['speeds']
    spikes = int(np.sum(speeds > spike_threshold)) if speeds.size else 0
    percent_time_moving = float(np.sum(speeds > motion_threshold) / (len(speeds)) ) if speeds.size else 0.0

    blink_rate = float(df['is_blinking'].sum()) / (duration+1e-6) if 'is_blinking' in df else 0.0
    
    # Gaze Dispersion logic will be handled in compute_saccade_metrics or here
    # We'll compute raw dispersion here for now
    gaze_dispersion = float(df['g_horizontal'].std(skipna=True)) if 'g_horizontal' in df else float('nan')

    features = {
        'duration_s': duration,
        'path_length_deg': path_len,
        'mean_angular_speed_deg_per_s': speed_stats['mean_speed'],
        'spike_count': spikes,
        'percent_time_moving': percent_time_moving,
        'blink_rate_per_s': blink_rate,
        'gaze_dispersion': gaze_dispersion
    }
    return features


def _select_gaze_series(df, label="Unknown", vel_thresh=0.8):
    """
    Selects the best gaze signal (calibrated pixels or raw ratio).
    Returns (times, signal, adjusted_vel_thresh, is_calibrated)
    """
    times = pd.to_numeric(df['t'], errors='coerce').to_numpy(dtype=float)
    
    # 1. Try Calibration
    model = load_calibration_model(label)
    is_calibrated = apply_calibration(df, model)
    
    if is_calibrated:
        print(f"Applied calibration model for {label}.")
        signal = df['gaze_x_px'].to_numpy()
        
        # Auto-scale threshold for pixels
        # If user passed a small ratio-like threshold (e.g. 0.8), scale it up
        if vel_thresh < 10.0:
            adjusted_thresh = 200.0 # Conservative pixel velocity threshold (lowered from 1000 to catch slower webcams)
            print(f"  -> Auto-scaling velocity threshold to {adjusted_thresh} px/s")
        else:
            adjusted_thresh = vel_thresh
    else:
        print(f"No calibration found for {label}. Using raw ratios.")
        # Fallback to raw columns
        candidate_cols = ['g_horizontal', 'left_px', 'right_px']
        signal = None
        for col in candidate_cols:
            if col in df.columns:
                col_values = pd.to_numeric(df[col], errors='coerce')
                if col_values.notna().sum() > 0:
                    signal = col_values.to_numpy(dtype=float)
                    break
        if signal is None:
            signal = np.zeros_like(times)
        adjusted_thresh = vel_thresh

    return times, signal, adjusted_thresh, is_calibrated


def _clean_numeric(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_clean_numeric(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_numeric(v) for k, v in value.items()}
    return value


def compute_saccade_metrics(
    csv_path,
    df=None,
    vel_thresh=0.8,
    min_dur=0.015,
    smooth_w=5,
    min_fix_dur=0.08,
    stimuli_times=None,
    interval_windows=None,
    latency_window=1.0,
):
    df = load_csv(csv_path) if df is None else df
    
    # Extract label from filename for calibration lookup
    filename = os.path.basename(csv_path)
    parts = filename.split('_')
    label = parts[0] if parts else "Unknown"

    times, pos, adjusted_thresh, is_calibrated = _select_gaze_series(df, label, vel_thresh)
    
    saccades = detect_saccades(times, pos, vel_thresh=adjusted_thresh, min_dur=min_dur, smooth_w=smooth_w)
    fixations = detect_fixations(times, pos, saccades, min_fix_dur=min_fix_dur)

    latencies = []
    if stimuli_times:
        latencies = saccade_latency_to_stimuli(saccades, stimuli_times, max_latency=latency_window)
        latencies = [lat if math.isfinite(lat) else None for lat in latencies]

    intrusive_total, intrusive_breakdown = (0, [])
    if interval_windows:
        intrusive_total, intrusive_breakdown = count_intrusive_saccades(saccades, interval_windows)

    # Re-calculate dispersion if calibrated to get pixel-based dispersion
    gaze_dispersion = float(np.std(pos)) if len(pos) > 0 else 0.0

    metrics = {
        'saccade_count': len(saccades),
        'fixation_count': len(fixations),
        'mean_saccade_duration_s': float(np.mean([s['duration'] for s in saccades])) if saccades else 0.0,
        'median_saccade_duration_s': float(np.median([s['duration'] for s in saccades])) if saccades else 0.0,
        'mean_saccade_peak_velocity': float(np.mean([s['peak_velocity'] for s in saccades])) if saccades else 0.0,
        'mean_saccade_amplitude': float(np.mean([s['amplitude'] for s in saccades])) if saccades else 0.0,
        'mean_fixation_duration_s': float(np.mean([f['duration'] for f in fixations])) if fixations else 0.0,
        'saccade_latencies_s': latencies,
        'intrusive_saccade_count': intrusive_total,
        'intrusive_counts_per_interval': intrusive_breakdown,
        'is_calibrated': is_calibrated,
        'gaze_dispersion': gaze_dispersion # Overwrite with calibrated version if available
    }
    return metrics


def extract_stimuli_from_csv(df):
    """
    Extracts stimulus onset times from the 'stimulus_state' column.
    Returns a list of timestamps where state transitions to 'target'.
    """
    if 'stimulus_state' not in df.columns:
        return []
    
    # Create a mask for 'target' state
    is_target = df['stimulus_state'] == 'target'
    
    # Find transitions: current is target, previous was not target
    # shift(1) gives previous value. fillna(False) handles the first row.
    transitions = is_target & (~is_target.shift(1).fillna(False))
    
    # Get times corresponding to these transitions
    times = df.loc[transitions, 't'].tolist()
    return times


def compute_summary(
    csv_path,
    spike_threshold=30.0,
    motion_threshold=5.0,
    vel_thresh=0.8,
    min_dur=0.015,
    smooth_w=5,
    min_fix_dur=0.08,
    stimuli_times=None,
    interval_windows=None,
    latency_window=1.0,
):
    df = load_csv(csv_path)
    
    # Fallback: Extract stimuli times from CSV if not provided
    if not stimuli_times:
        stimuli_times = extract_stimuli_from_csv(df)
        if stimuli_times:
            print(f"Extracted {len(stimuli_times)} stimuli onsets from CSV.")

    combined = compute_features(
        csv_path,
        spike_threshold=spike_threshold,
        motion_threshold=motion_threshold,
        df=df,
    )
    saccade_metrics = compute_saccade_metrics(
        csv_path,
        df=df,
        vel_thresh=vel_thresh,
        min_dur=min_dur,
        smooth_w=smooth_w,
        min_fix_dur=min_fix_dur,
        stimuli_times=stimuli_times,
        interval_windows=interval_windows,
        latency_window=latency_window,
    )
    combined.update(saccade_metrics)
    return _clean_numeric(combined)


def _load_scalar_list(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        
        # FIX: Handle session summary dicts (e.g. {"stimuli_directions": [...]})
        if isinstance(data, dict):
            # Look for the list inside known keys
            found_list = None
            for key in ['stimuli_directions', 'stimuli_times', 'stimuli']:
                if key in data and isinstance(data[key], list):
                    found_list = data[key]
                    break
            
            if found_list is not None:
                data = found_list
            else:
                # If it's a dict but has no stimuli list, return empty
                return []

        # Handle list of dicts (e.g. stimuli_directions) or list of floats
        if isinstance(data, list):
            values = []
            for item in data:
                if isinstance(item, dict):
                    # Try to find a time-like key
                    # 'time' is used in stimuli_directions
                    val = item.get('time', item.get('timestamp', item.get('onset', 0)))
                    values.append(float(val))
                else:
                    values.append(float(item))
            return values
            
    except json.JSONDecodeError:
        pass
    
    # Fallback: Line-based parsing (CSV-like)
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split(',')[0].strip()
        try:
            values.append(float(token))
        except ValueError:
            # Skip lines that aren't numbers (like JSON braces)
            continue
    return values

def _load_interval_list(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        intervals = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and 'start' in entry and 'end' in entry:
                    intervals.append((float(entry['start']), float(entry['end'])))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    intervals.append((float(entry[0]), float(entry[1])))
        if intervals:
            return intervals
    except json.JSONDecodeError:
        pass
    parsed_intervals = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
        if len(parts) >= 2:
            parsed_intervals.append((float(parts[0]), float(parts[1])))
    return parsed_intervals


def _append_dict_to_csv(path, data):
    path = Path(path)
    write_header = not path.exists()
    fieldnames = list(data.keys())
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute gaze/head metrics and saccade statistics for a session CSV.')
    parser.add_argument('csv_path', help='Path to session CSV produced by collect_data.py')
    parser.add_argument('--spike-threshold', type=float, default=30.0, help='Angular speed threshold for counting spikes (deg/s).')
    parser.add_argument('--motion-threshold', type=float, default=5.0, help='Angular speed threshold for percent_time_moving (deg/s).')
    parser.add_argument('--vel-thresh', type=float, default=0.8, help='Velocity threshold for saccade detection (gaze-units/s).')
    parser.add_argument('--min-saccade-dur', type=float, default=0.015, help='Minimum saccade duration in seconds.')
    parser.add_argument('--smooth-window', type=int, default=5, help='Window size for moving-average smoothing before velocity calc.')
    parser.add_argument('--min-fix-dur', type=float, default=0.08, help='Minimum fixation duration in seconds.')
    parser.add_argument('--latency-window', type=float, default=2.0, help='Maximum latency window (s) when pairing stimuli to saccades.')
    parser.add_argument('--stimuli-file', help='Optional path to JSON or newline file listing stimulus onset timestamps (seconds).')
    parser.add_argument('--intervals-file', help='Optional path to JSON or newline file listing intrusive-interval pairs (start,end).')
    parser.add_argument('--out', help='Write summary JSON to this path.')
    parser.add_argument('--csv-out', help='Append the summary as a CSV row at this path.')

    args = parser.parse_args()

    stimuli_time_values = _load_scalar_list(args.stimuli_file) if args.stimuli_file else None
    interval_window_values = _load_interval_list(args.intervals_file) if args.intervals_file else None

    summary_payload = compute_summary(
        args.csv_path,
        spike_threshold=args.spike_threshold,
        motion_threshold=args.motion_threshold,
        vel_thresh=args.vel_thresh,
        min_dur=args.min_saccade_dur,
        smooth_w=args.smooth_window,
        min_fix_dur=args.min_fix_dur,
        stimuli_times=stimuli_time_values,
        interval_windows=interval_window_values,
        latency_window=args.latency_window,
    )

    print(json.dumps(summary_payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    if args.csv_out:
        _append_dict_to_csv(args.csv_out, summary_payload)