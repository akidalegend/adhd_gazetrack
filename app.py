import argparse
import json
from pathlib import Path

from run_calibration import (
    _wait_for_click,
    collect_calibration_points,
    compute_calibration_model,
    get_screen_resolution,
    verify_calibration,
)
from task_utils import ensure_directories, prompt_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable launcher for calibration (no fullscreen)")
    parser.add_argument("--label", help="Participant label")
    parser.add_argument("--stimulus-width", type=int, default=1400, help="Stimulus window width in pixels")
    parser.add_argument("--stimulus-height", type=int, default=900, help="Stimulus window height in pixels")
    parser.add_argument(
        "--window-x",
        type=int,
        default=0,
        help="Top-left X position for stimulus window (set to monitor origin)",
    )
    parser.add_argument(
        "--window-y",
        type=int,
        default=0,
        help="Top-left Y position for stimulus window (set to monitor origin)",
    )
    parser.add_argument(
        "--start-width",
        type=int,
        default=600,
        help="Start-screen width for the click prompt",
    )
    parser.add_argument(
        "--start-height",
        type=int,
        default=400,
        help="Start-screen height for the click prompt",
    )
    args = parser.parse_args()

    label = args.label if args.label else prompt_label()
    ensure_directories(["sessions/calibration"])

    print("Ensure the user is sitting ~60cm from the screen and lighting is consistent.")
    print("Click the start window to begin calibration or press q to cancel.")

    if not _wait_for_click(width=args.start_width, height=args.start_height):
        print("Calibration cancelled before start click.")
        return

    screen_w, screen_h = get_screen_resolution(args.stimulus_width, args.stimulus_height)

    cal_data = collect_calibration_points(
        label,
        screen_w,
        screen_h,
        args.window_x,
        args.window_y,
        False,  # fullscreen off for stability
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

    # Verification uses the same stable window sizing (no fullscreen)
    verify_calibration(model, screen_w, screen_h, args.window_x, args.window_y, False)


if __name__ == "__main__":
    main()
