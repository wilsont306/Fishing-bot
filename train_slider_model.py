"""Train the slider bot's learned dynamics model from recorded CSV data.

Usage:
    python train_slider_model.py slider_training_data.csv [output.npz]

The output slider_model_v4.npz should stay beside slider_bot.py. The
trainer uses only NumPy, so it does not require scikit-learn or a GPU.
"""

import csv
import collections
import os
import sys

import numpy as np


RIDGE = 12.0
MAX_SAMPLES_PER_RUN_ACTION = 800


def load_transitions(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped_rows = collections.defaultdict(list)
    for row in rows:
        grouped_rows[row["run_id"]].append(row)
    accepted_runs = set()
    rejected_runs = []
    for run_id, run_rows in grouped_rows.items():
        times = np.asarray([float(row["time"]) for row in run_rows])
        errors = np.asarray([abs(float(row["position_error"]))
                             for row in run_rows])
        holding = np.asarray([int(row["holding"]) for row in run_rows])
        duration = times[-1] - times[0] if len(times) > 1 else 0.0
        fps = len(run_rows) / duration if duration > 0 else 0.0
        reason = None
        if len(run_rows) < 60:
            reason = "too short"
        elif not 4.0 <= fps <= 30.0:
            reason = f"invalid capture rate ({fps:.1f} FPS)"
        elif np.median(errors) > 120.0:
            reason = f"bad detection (median error {np.median(errors):.0f}px)"
        elif not 0.02 <= holding.mean() <= 0.98:
            reason = "input stuck in one state"
        if reason:
            rejected_runs.append((run_id, reason))
        else:
            accepted_runs.add(run_id)
    if rejected_runs:
        print(f"Quarantined {len(rejected_runs)} bad recording session(s):")
        for run_id, reason in rejected_runs:
            print(f"  {run_id}: {reason}")

    by_run = {0: collections.defaultdict(list),
              1: collections.defaultdict(list)}
    # Keep switch-onset samples so the model learns the one/two-frame actuator
    # delay instead of assuming a click changes acceleration immediately.
    for index in range(2, len(rows) - 2):
        current = rows[index]
        if current["run_id"] not in accepted_runs:
            continue
        following = rows[index + 2]
        previous = rows[index - 1]
        before_previous = rows[index - 2]
        window = rows[index - 2:index + 3]
        if len({item["run_id"] for item in window}) != 1:
            continue
        action = int(current["holding"])
        if int(rows[index + 1]["holding"]) != action \
                or int(following["holding"]) != action:
            continue
        dt = float(following["time"]) - float(current["time"])
        step_dt = dt / 2.0
        if not 0.025 <= step_dt <= 0.18:
            continue
        slider_now = float(current["slider_center"])
        slider_next = float(following["slider_center"])
        box_now = float(current["box_center"])
        box_next = float(following["box_center"])
        if abs(slider_next - slider_now) > 80 or abs(box_next - box_now) > 55:
            continue
        velocity = float(current["slider_velocity"])
        next_velocity = float(following["slider_velocity"])
        previous_velocity = float(previous["slider_velocity"])
        recent_dt = float(previous["time"]) - float(before_previous["time"])
        if not 0.020 <= recent_dt <= 0.18:
            continue
        recent_accel = ((previous_velocity
                         - float(before_previous["slider_velocity"]))
                        / recent_dt)
        acceleration = (next_velocity - velocity) / dt
        target_velocity = float(current["target_velocity"])
        if (not np.isfinite(acceleration)
                or not np.isfinite(recent_accel)
                or abs(acceleration) > 5000
                or abs(velocity) > 340
                or abs(target_velocity) > 180):
            continue
        box_height = float(current["box_bottom"]) - float(current["box_top"]) + 1
        slider_height = (float(current["slider_bottom"])
                         - float(current["slider_top"]) + 1)
        if not 15 <= box_height <= 70 or not 5 <= slider_height <= 55:
            continue
        previous_action = int(previous["holding"])
        # Stable commands must have the known direction. Switch-onset samples
        # may briefly retain the previous direction and are kept so the model
        # learns real actuator delay.
        if previous_action == action:
            if (action == 0 and acceleration <= 50.0) or \
                    (action == 1 and acceleration >= -50.0):
                continue
        feature = [
            velocity,
            float(previous_action),
            float(np.clip(recent_accel, -5000.0, 5000.0)),
            step_dt,
        ]
        by_run[action][current["run_id"]].append((feature, acceleration))

    features = {0: [], 1: []}
    targets = {0: [], 1: []}
    for action in (0, 1):
        for run_id, samples in by_run[action].items():
            # A 10,000-row poor session must not overpower every good session.
            if len(samples) > MAX_SAMPLES_PER_RUN_ACTION:
                indexes = np.linspace(0, len(samples) - 1,
                                      MAX_SAMPLES_PER_RUN_ACTION,
                                      dtype=int)
                samples = [samples[index] for index in indexes]
            for feature, target in samples:
                features[action].append(feature)
                targets[action].append(target)
    return features, targets


def fit_ridge(x, y, mean, scale, weights):
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    root_w = np.sqrt(weights)[:, None]
    weighted_design = design * root_w
    weighted_y = y * root_w[:, 0]
    return np.linalg.solve(weighted_design.T @ weighted_design + penalty,
                           weighted_design.T @ weighted_y)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python train_slider_model_v4.py slider_training_data.csv [output.npz]")
    input_path = os.path.abspath(sys.argv[1])
    output_path = (os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3
                   else os.path.join(os.path.dirname(input_path),
                                     "slider_model_v4.npz"))

    features, targets = load_transitions(input_path)
    if len(features[0]) < 150 or len(features[1]) < 150:
        raise SystemExit(
            "Not enough clean stable-action data. Record another 2-5 minutes "
            "so both hold and release have at least 150 clean samples."
        )

    all_x = np.asarray(features[0] + features[1], dtype=float)
    mean = all_x.mean(axis=0)
    scale = all_x.std(axis=0)
    scale[scale < 1e-6] = 1.0

    coefficients = {}
    for action, name in ((0, "release"), (1, "hold")):
        x = np.asarray(features[action], dtype=float)
        y = np.asarray(targets[action], dtype=float)
        # Trim extreme detector/timing noise without assuming a fixed physics model.
        low, high = np.percentile(y, [2.0, 98.0])
        keep = (y >= low) & (y <= high)
        x, y = x[keep], y[keep]
        # Iteratively reweighted Huber fit: detector/timing outliers lose
        # influence without allowing poor sessions to move the entire model.
        weights = np.ones(len(y), dtype=float)
        coef = fit_ridge(x, y, mean, scale, weights)
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        for _ in range(4):
            residual = y - design @ coef
            sigma = max(25.0, 1.4826 * np.median(np.abs(residual)))
            cutoff = 1.5 * sigma
            weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-9))
            coef = fit_ridge(x, y, mean, scale, weights)
        rmse = float(np.sqrt(np.mean((design @ coef - y) ** 2)))
        coefficients[action] = coef
        print(f"{name}: {len(y)} samples, acceleration RMSE={rmse:.1f} px/s^2")

    np.savez(
        output_path,
        feature_mean=mean,
        feature_scale=scale,
        release_coef=coefficients[0],
        hold_coef=coefficients[1],
        feature_names=np.array([
            "slider_velocity", "previous_holding",
            "recent_acceleration", "frame_dt"
        ]),
        format_version=np.array([4], dtype=np.int32),
    )
    print(f"Saved model: {output_path}")


if __name__ == "__main__":
    main()
