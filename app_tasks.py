import argparse
from pathlib import Path

from run_calibration import (
    _wait_for_click as calibration_wait,
    collect_calibration_points,
    compute_calibration_model,
    get_screen_resolution,
    verify_calibration,
)
from run_fixation_task import run_fixation_task
from run_prosaccade_task import run_trials as run_prosaccade_trials
from run_antisaccade_task import run_trials as run_antisaccade_trials
from task_utils import ensure_dir, ensure_directories, prompt_label
import json


def run_calibration_mode(label: str, args) -> None:
    ensure_directories(["sessions/calibration"])
    print("Ensure ~60cm distance, steady head, even lighting.")
    print("Click the start window to begin calibration or press q to cancel.")

    if not calibration_wait(width=args.start_width, height=args.start_height):
        print("Calibration cancelled before start click.")
        return

    screen_w, screen_h = get_screen_resolution(args.stimulus_width, args.stimulus_height)
    cal_data = collect_calibration_points(
        label,
        screen_w,
        screen_h,
        args.window_x,
        args.window_y,
        False,
    )
    if not cal_data:
        print("Calibration aborted.")
        return

    model = compute_calibration_model(cal_data)
    if not model:
        print("Calibration failed to generate a model.")
        return

    cal_data["model"] = model
    print("\nCalibration Successful!")
    print(f"X Model: ScreenX = {model['x_slope']:.2f} * GazeH + {model['x_intercept']:.2f}")
    print(f"Y Model: ScreenY = {model['y_slope']:.2f} * GazeV + {model['y_intercept']:.2f}")

    filename = Path("sessions/calibration") / f"{label}_calibration.json"
    with filename.open("w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=4)
    print(f"Saved to {filename}")

    verify_calibration(model, screen_w, screen_h, args.window_x, args.window_y, False)


def run_fixation_mode(label: str, args) -> None:
    raw_path = run_fixation_task(
        label=label,
        duration=args.duration,
        master_csv=Path(args.master_csv),
        raw_dir=Path(args.raw_dir),
        summary_dir=Path(args.summary_dir) if args.summary_dir else None,
        show_dot=not args.no_dot,
        dot_size=args.dot_size,
        dot_radius=args.dot_radius,
    )
    print(f"Fixation task complete. Raw session saved at {raw_path}. Metrics appended to {args.master_csv}.")


def run_prosaccade_mode(label: str, args) -> None:
    raw_path = run_prosaccade_trials(
        label=label,
        trials=args.trials,
        center_duration=args.center_duration,
        gap_duration=args.gap_duration,
        target_duration=args.target_duration,
        stimulus_size=args.stimulus_size,
        center_radius=args.center_radius,
        target_radius=args.target_radius,
        offset_ratio=args.offset_ratio,
        master_csv=Path(args.master_csv),
        raw_dir=Path(args.raw_dir),
        summary_dir=Path(args.summary_dir) if args.summary_dir else None,
    )
    print(f"Prosaccade task complete. Raw session saved at {raw_path}. Metrics appended to {args.master_csv}.")


def run_antisaccade_mode(label: str, args) -> None:
    raw_path = run_antisaccade_trials(
        label=label,
        trials=args.trials,
        center_duration=args.center_duration,
        gap_duration=args.gap_duration,
        target_duration=args.target_duration,
        stimulus_size=args.stimulus_size,
        center_radius=args.center_radius,
        target_radius=args.target_radius,
        offset_ratio=args.offset_ratio,
        master_csv=Path(args.master_csv),
        raw_dir=Path(args.raw_dir),
        summary_dir=Path(args.summary_dir) if args.summary_dir else None,
    )
    print(f"Antisaccade task complete. Raw session saved at {raw_path}. Metrics appended to {args.master_csv}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified app launcher for calibration and tasks (stable windows)")
    parser.add_argument("--mode", choices=["calibration", "fixation", "prosaccade", "antisaccade"], default="calibration")
    parser.add_argument("--label", help="Participant/session label")
    # Window controls for calibration
    parser.add_argument("--stimulus-width", type=int, default=1400, help="Stimulus window width in pixels (calibration)")
    parser.add_argument("--stimulus-height", type=int, default=900, help="Stimulus window height in pixels (calibration)")
    parser.add_argument("--window-x", type=int, default=0, help="Top-left X for calibration window")
    parser.add_argument("--window-y", type=int, default=0, help="Top-left Y for calibration window")
    parser.add_argument("--start-width", type=int, default=600, help="Start-screen width (calibration)")
    parser.add_argument("--start-height", type=int, default=400, help="Start-screen height (calibration)")
    # Shared metrics destinations
    parser.add_argument("--master-csv", default="master_metrics.csv", help="Path to cumulative metrics CSV")
    parser.add_argument("--raw-dir", default="sessions/raw", help="Directory to store raw per-frame CSV files")
    parser.add_argument("--summary-dir", default="sessions/summaries", help="Directory for per-session JSON summaries")
    # Fixation-specific
    parser.add_argument("--duration", type=float, default=25.0, help="Fixation duration (s)")
    parser.add_argument("--no-dot", action="store_true", help="Disable fixation dot window")
    parser.add_argument("--dot-size", type=int, default=700, help="Fixation dot window size (px)")
    parser.add_argument("--dot-radius", type=int, default=18, help="Fixation dot radius (px)")
    # Saccade task settings
    parser.add_argument("--trials", type=int, default=20, help="Number of saccade trials")
    parser.add_argument("--center-duration", type=float, default=1.0, help="Seconds to show center fixation per trial")
    parser.add_argument("--gap-duration", type=float, default=0.2, help="Gap (s) between center offset and target onset")
    parser.add_argument("--target-duration", type=float, default=1.5, help="Seconds to keep target visible")
    parser.add_argument("--stimulus-size", type=int, default=600, help="Stimulus window size (px) for saccade tasks")
    parser.add_argument("--center-radius", type=int, default=12, help="Center dot radius (px)")
    parser.add_argument("--target-radius", type=int, default=14, help="Target dot radius (px)")
    parser.add_argument("--offset-ratio", type=float, default=0.35, help="Offset from center as ratio of window width (0-0.5)")

    args = parser.parse_args()
    label = args.label if args.label else prompt_label()

    if args.mode == "calibration":
        run_calibration_mode(label, args)
    elif args.mode == "fixation":
        ensure_dir(Path(args.raw_dir))
        if args.summary_dir:
            ensure_dir(Path(args.summary_dir))
        run_fixation_mode(label, args)
    elif args.mode == "prosaccade":
        ensure_dir(Path(args.raw_dir))
        if args.summary_dir:
            ensure_dir(Path(args.summary_dir))
        run_prosaccade_mode(label, args)
    elif args.mode == "antisaccade":
        ensure_dir(Path(args.raw_dir))
        if args.summary_dir:
            ensure_dir(Path(args.summary_dir))
        run_antisaccade_mode(label, args)


if __name__ == "__main__":
    main()
