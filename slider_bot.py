"""Predictive computer-vision controller for a vertical slider minigame.

Controls:
  F9  - Mark each of the two opposite capture-region corners
  F7  - Start or pause the controller
  F8  - Panic stop, release inputs, and exit
  F10 - Save a diagnostic snapshot
  F11 - Show or hide the live overlay

Optional arguments:
  --debug                Print live controller diagnostics
  --record [FILE.csv]    Append accepted observations to a training CSV
  --model FILE.npz       Load a specific compatible model
"""

import sys
import time
import threading
import csv
import os
import glob
import math

import numpy as np
import mss
from pynput.mouse import Controller, Button
from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Controller as KeyboardController

BOT_VERSION = "v15.6-target-pattern-prediction"

ONLINE_FORGETTING = 0.9995
ONLINE_INITIAL_COVARIANCE = 0.00025
ONLINE_RESIDUAL_CLIP = 250.0
ONLINE_SAVE_EVERY = 100
ONLINE_MAX_COEF_DRIFT = np.array([25.0, 25.0, 25.0, 20.0, 12.0])

# ---------------------------------------------------------------------------
# Configuration - tweak these if detection isn't reliable for your game
# ---------------------------------------------------------------------------
# All ranges are BGR (mss gives BGRA, alpha is dropped).
# Use color_picker.py to find your game's real values and replace these.

# Confirmed by sampling actual screenshots of this game's UI:
#   yellow border = pure RGB(255,255,0)  -> BGR(0,255,255)
#   green border   = pure RGB(0,255,0)    -> BGR(0,255,0)
#   white slider   = pure RGB(255,255,255)-> BGR(255,255,255)
# Ranges below are tight (small tolerance for JPEG/compression artifacts and
# anti-aliased edge pixels), not loose guesses, since the source colors are
# fully saturated and very distinct from the rest of the UI.

YELLOW_LOWER = np.array([0, 200, 200])    # BGR - low blue, high green, high red
YELLOW_UPPER = np.array([60, 255, 255])

GREEN_LOWER = np.array([0, 200, 0])       # BGR - low blue, high green, low red
GREEN_UPPER = np.array([60, 255, 60])

WHITE_LOWER = np.array([235, 235, 235])
WHITE_UPPER = np.array([255, 255, 255])

# The control loop no longer throttles itself - it runs as fast as screen
# capture allows for minimum reaction latency. This uses more CPU; that's
# the tradeoff for responsiveness.
POLL_HZ = 60  # retained for reference only; does not limit the loop
DEAD_ZONE_PX = 4  # ignore tiny overlaps/gaps to reduce click jitter
EDGE_SAFETY_MARGIN_PX = 2.0  # soft inset used by ML trajectory cost
EDGE_OVERRIDE_MIN_SPEED_PX_S = 40.0

# ---- OVERSHOOT COMPENSATION (the main knobs to tune) ----
# The slider has momentum. Deciding based on where it is RIGHT NOW means the
# release always happens too late and it sails past the target.
#
# The two directions need DIFFERENT values because the physics aren't
# symmetric: holding applies a controlled upward force, but releasing lets
# gravity accelerate the slider downward, and that acceleration keeps
# building. Falling therefore needs more anticipation than rising.
#   too LOW  -> still overshoots in that direction
#   too HIGH -> reacts early, undershoots or oscillates
# Adjust in steps of ~0.01.
LOOKAHEAD_UP_S = 0.060   # rising: release earlier before momentum carries past target
LOOKAHEAD_DOWN_S = 0.070 # falling: begin holding earlier against relative descent

# Adaptive physics controller. It learns the slider's acceleration under each
# mouse state and uses v^2/(2a) stopping distance instead of one fixed horizon.
CONTROL_LATENCY_S = 0.025
INITIAL_UP_ACCEL_PX_S2 = 1200.0
INITIAL_DOWN_ACCEL_PX_S2 = 1200.0
ACCEL_LEARNING_RATE = 0.08
MIN_LEARNED_ACCEL_PX_S2 = 650.0
MAX_LEARNED_ACCEL_PX_S2 = 3500.0
MAX_STOPPING_DISTANCE_PX = 42.0

# Bounds learned from the supplied logs.  One-frame shape changes otherwise
# create 500+ px/s velocity spikes and cause the large overshoots seen in v8.
MAX_TRACKED_SPEED_PX_S = 320.0
MAX_TARGET_SPEED_PX_S = 160.0
TARGET_VELOCITY_SMOOTHING = 0.70
TARGET_ACCELERATION_SMOOTHING = 0.65
MAX_TARGET_ACCEL_PX_S2 = 2400.0
TARGET_REVERSAL_SPEED_PX_S = 28.0
VELOCITY_WARMUP_FRAMES = 2
MAX_VALID_FRAME_DT_S = 0.20
MAX_FRAME_JUMP_PX = 32.0
MAX_TRACK_X_DRIFT_PX = 24.0
MAX_TARGET_WIDTH_PX = 80.0

# Smooths the velocity estimate so one noisy frame can't produce a wild
# prediction. 0.0 = no smoothing (twitchy), 0.9 = very heavy (sluggish).
VELOCITY_SMOOTHING = 0.6

# Below this speed (px/sec) the slider is treated as stalled - typically
# pinned against the top or bottom of the track - and the velocity estimate
# snaps to the measured value instead of being smoothed. Without this, a
# stale velocity keeps being projected forward at the track limits.
STALL_VELOCITY_PX_S = 40.0

# ---- IDLE CLICK ----
# Once the bot has seen the minigame at least once, going this long with no
# detection is treated as "the minigame ended" and a single click is sent
# (useful for starting the next one). Nothing is clicked before the first
# successful detection, so the bot never clicks blindly on startup.
#
# Exactly ONE click is sent per idle spell. After clicking, the bot waits
# for a genuine detection before it will click again, so a long gap can
# never turn into repeated clicking.
IDLE_CLICK_AFTER_S = 2.0
HARD_CLICK_AFTER_S = 20.0

# Before the idle click, hold this key down for this long. Note this blocks
# the control loop for the duration, which is fine because it only runs
# while detection is already lost.
PRE_CLICK_KEY = "t"
PRE_CLICK_HOLD_S = 2.5
POST_CLICK_T_HOLD_S = 0.20
PRE_CLICK_RELEASE_DELAY_S = 0.12

mouse = Controller()
keyboard_ctrl = KeyboardController()


class LearnedDynamicsModel:
    """State-aware learned dynamics used by the predictive controller."""

    def __init__(self, path):
        data = np.load(path)
        self.mean = data["feature_mean"]
        self.scale = data["feature_scale"]
        self.hold_coef = data["hold_coef"]
        self.release_coef = data["release_coef"]
        version = int(np.ravel(data.get("format_version", [0]))[0])
        if version not in (4, 5) or len(self.mean) != 4:
            raise ValueError(
                "old/incompatible model format; use slider_model_v5_online.npz "
                "or seed it from the supplied v4 model"
            )
        self.path = path
        size = len(self.mean) + 1
        self.release_cov = np.array(
            data["release_cov"] if "release_cov" in data
            else np.eye(size) * ONLINE_INITIAL_COVARIANCE, dtype=float)
        self.hold_cov = np.array(
            data["hold_cov"] if "hold_cov" in data
            else np.eye(size) * ONLINE_INITIAL_COVARIANCE, dtype=float)
        self.sample_counts = np.array(
            data["sample_counts"] if "sample_counts" in data
            else [0, 0], dtype=np.int64)
        self.accepted_since_save = 0
        self.dirty = version == 4
        self.reference_release_coef = self.release_coef.copy()
        self.reference_hold_coef = self.hold_coef.copy()
        # Models created by v15.0-v15.2 used a larger covariance and adapted
        # too aggressively. Cap it immediately when loading an older online file.
        for covariance in (self.release_cov, self.hold_cov):
            max_diag = float(np.max(np.diag(covariance)))
            if max_diag > ONLINE_INITIAL_COVARIANCE:
                covariance *= ONLINE_INITIAL_COVARIANCE / max_diag
                self.dirty = True

    def set_safety_reference(self, path):
        """Anchor online learning to the matching offline v4 model."""
        reference = np.load(path)
        if int(np.ravel(reference.get("format_version", [0]))[0]) != 4:
            raise ValueError("safety reference is not a v4 model")
        if (not np.allclose(reference["feature_mean"], self.mean,
                            rtol=1e-6, atol=1e-6)
                or not np.allclose(reference["feature_scale"], self.scale,
                                   rtol=1e-6, atol=1e-6)):
            raise ValueError("v4 reference was trained from a different feature scale")
        self.reference_release_coef = np.array(reference["release_coef"],
                                               dtype=float)
        self.reference_hold_coef = np.array(reference["hold_coef"], dtype=float)
        before_release = self.release_coef.copy()
        before_hold = self.hold_coef.copy()
        self.release_coef[:] = np.clip(
            self.release_coef,
            self.reference_release_coef - ONLINE_MAX_COEF_DRIFT,
            self.reference_release_coef + ONLINE_MAX_COEF_DRIFT)
        self.hold_coef[:] = np.clip(
            self.hold_coef,
            self.reference_hold_coef - ONLINE_MAX_COEF_DRIFT,
            self.reference_hold_coef + ONLINE_MAX_COEF_DRIFT)
        correction = max(float(np.max(np.abs(self.release_coef - before_release))),
                         float(np.max(np.abs(self.hold_coef - before_hold))))
        if correction > 1e-9:
            self.dirty = True
        return correction

    def acceleration(self, velocity, previous_hold, recent_accel,
                     measured_dt, hold):
        x = np.array([
            velocity, float(previous_hold), recent_accel, measured_dt
        ], dtype=float)
        z = (x - self.mean) / self.scale
        design = np.concatenate(([1.0], z))
        coef = self.hold_coef if hold else self.release_coef
        prediction = float(np.clip(design @ coef, -5000.0, 5000.0))
        # Once the command is stable, enforce its known physical direction.
        # On the first frame after a switch, allow the old momentum/actuator
        # delay to continue briefly; v3 incorrectly assumed instant response.
        if bool(previous_hold) == bool(hold):
            return min(-50.0, prediction) if hold else max(50.0, prediction)
        return prediction

    def observe(self, velocity, previous_hold, recent_accel, measured_dt,
                action, measured_accel, stable_action_frames):
        """Safely update one action model from a verified live transition."""
        if stable_action_frames < 3:
            return False, None
        if not 0.025 <= measured_dt <= MAX_VALID_FRAME_DT_S:
            return False, None
        if abs(velocity) > 340 or abs(recent_accel) > 5000:
            return False, None
        if not np.isfinite(measured_accel) or abs(measured_accel) > 5000:
            return False, None
        # Stable HOLD must accelerate upward; stable RELEASE downward. Flat
        # samples at hard track limits and detector glitches are not learned.
        if (action and measured_accel >= -50.0) or \
                (not action and measured_accel <= 50.0):
            return False, None

        feature = np.array([
            velocity, float(previous_hold), recent_accel, measured_dt
        ], dtype=float)
        z = (feature - self.mean) / self.scale
        design = np.concatenate(([1.0], z))
        coef = self.hold_coef if action else self.release_coef
        covariance = self.hold_cov if action else self.release_cov
        prediction = float(design @ coef)
        residual = float(np.clip(measured_accel - prediction,
                                 -ONLINE_RESIDUAL_CLIP,
                                 ONLINE_RESIDUAL_CLIP))
        denominator = ONLINE_FORGETTING + design @ covariance @ design
        gain = covariance @ design / denominator
        coef += gain * residual
        reference_coef = (self.reference_hold_coef if action
                          else self.reference_release_coef)
        coef[:] = np.clip(coef,
                          reference_coef - ONLINE_MAX_COEF_DRIFT,
                          reference_coef + ONLINE_MAX_COEF_DRIFT)
        covariance[:] = ((covariance - np.outer(gain, design) @ covariance)
                         / ONLINE_FORGETTING)
        covariance[:] = 0.5 * (covariance + covariance.T)
        np.clip(coef, -2500.0, 2500.0, out=coef)
        self.sample_counts[int(bool(action))] += 1
        self.accepted_since_save += 1
        self.dirty = True
        return True, residual

    def save(self, force=False):
        """Atomically persist online state without touching older models."""
        if not self.dirty:
            return False
        if not force and self.accepted_since_save < ONLINE_SAVE_EVERY:
            return False
        temporary = self.path + ".tmp.npz"
        np.savez(
            temporary,
            feature_mean=self.mean,
            feature_scale=self.scale,
            release_coef=self.release_coef,
            hold_coef=self.hold_coef,
            release_cov=self.release_cov,
            hold_cov=self.hold_cov,
            sample_counts=self.sample_counts,
            feature_names=np.array([
                "slider_velocity", "previous_holding",
                "recent_acceleration", "frame_dt"
            ]),
            format_version=np.array([5], dtype=np.int32),
        )
        os.replace(temporary, self.path)
        self.accepted_since_save = 0
        self.dirty = False
        return True

    def choose_action(self, error, velocity, target_velocity,
                      target_acceleration,
                      currently_holding, allowed_error, recent_accel,
                      measured_dt):
        """Evaluate stable and one-switch trajectories, then replan next frame."""
        # The real loop commonly runs at 8-12 FPS, not 60 FPS. Simulating a
        # new decision every 16 ms made the controller depend on corrections
        # it physically could not issue until 80-125 ms later. Use the actual
        # observed capture interval and keep a consistent time horizon.
        dt = max(0.025, min(0.125, measured_dt or 0.080))
        horizon = max(3, min(8, int(math.ceil(0.42 / dt))))
        best_cost = float("inf")
        best_first = currently_holding
        best_path = []
        policies = []
        for first in (False, True):
            policies.append([first] * horizon)
            for switch_at in range(1, horizon):
                policies.append([first] * switch_at
                                + [not first] * (horizon - switch_at))
        for policy in policies:
            sim_error = float(error)
            sim_velocity = float(velocity)
            sim_target_velocity = float(target_velocity)
            cost = 0.0
            previous_action = currently_holding
            sim_accel = float(recent_accel)
            path = []
            for step, action in enumerate(policy):
                accel = self.acceleration(sim_velocity, previous_action,
                                          sim_accel, dt, action)
                sim_velocity += accel * dt
                sim_target_velocity += target_acceleration * dt
                sim_target_velocity = max(-MAX_TARGET_SPEED_PX_S,
                                          min(MAX_TARGET_SPEED_PX_S,
                                              sim_target_velocity))
                sim_error += (sim_velocity - sim_target_velocity) * dt
                path.append(sim_error)
                outside = max(0.0, abs(sim_error) - allowed_error)
                cost += (outside * outside * 28.0
                         + outside * outside * outside * 2.0
                         + sim_error * sim_error * 0.04)
                if action != previous_action:
                    cost += 1.5
                previous_action = action
                sim_accel = accel
                if step == 0 and action != currently_holding:
                    cost += 0.5
            terminal_outside = max(0.0, abs(sim_error) - allowed_error)
            cost += terminal_outside * terminal_outside * 32.0
            if cost < best_cost:
                best_cost = cost
                best_first = policy[0]
                best_path = path
        return bool(best_first), best_cost, best_path


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def capture_corner(state, which):
    """Record one corner of the region at the current mouse position.

    Hotkey-driven so it never touches stdin - hover the corner in the game
    and tap F9. No console interaction, no input(), nothing that can EOF.
    """
    x, y = mouse.position
    if which == "first":
        state.corner1 = (x, y)
        print(f"Corner 1 set at ({x}, {y}). Now hover the OPPOSITE corner and press F9 again.")
    else:
        state.corner2 = (x, y)
        x1, y1 = state.corner1
        x2, y2 = state.corner2
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)

        if width < 5 or height < 20:
            print("Region too small - press F9 twice again to retry.")
            state.corner1 = None
            state.corner2 = None
            return

        state.region = {"left": int(left), "top": int(top),
                        "width": int(width), "height": int(height)}
        print(f"Region set: {state.region}")
        print("Press F7 to start.")
        state.corner1 = None
        state.corner2 = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def grab(sct, region):
    frame = np.array(sct.grab(region))
    return frame[:, :, :3]  # BGR


def rows_matching(frame_bgr, lower, upper):
    mask = np.all((frame_bgr >= lower) & (frame_bgr <= upper), axis=2)
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return int(rows.min()), int(rows.max())


def color_bbox(frame_bgr, lower, upper, min_height=12, max_height=70):
    """Return the strongest realistic color object's (top,bottom,left,right).

    Unlike a global min/max, this keeps unrelated same-color UI pixels from
    being merged with the target when the selected capture is very wide.
    """
    mask = np.all((frame_bgr >= lower) & (frame_bgr <= upper), axis=2)
    active_rows = np.nonzero(mask.any(axis=1))[0]
    if not len(active_rows):
        return None
    runs = []
    start = prev = active_rows[0]
    for row in active_rows[1:]:
        if row == prev + 1:
            prev = row
        else:
            runs.append((start, prev))
            start = prev = row
    runs.append((start, prev))

    candidates = []
    for top, bottom in runs:
        height = bottom - top + 1
        if not min_height <= height <= max_height:
            continue
        section = mask[top:bottom + 1]
        cols = np.nonzero(section.any(axis=0))[0]
        if not len(cols):
            continue
        # The target's top/bottom border joins its columns into one run.
        col_runs = []
        left = right = cols[0]
        for col in cols[1:]:
            if col <= right + 2:
                right = col
            else:
                col_runs.append((left, right))
                left = right = col
        col_runs.append((left, right))
        for left, right in col_runs:
            width = right - left + 1
            if 5 <= width <= 180:
                score = int(section[:, left:right + 1].sum())
                candidates.append((score, top, bottom, left, right))
    if not candidates:
        return None
    _, top, bottom, left, right = max(candidates)
    return int(top), int(bottom), int(left), int(right)


def find_box(frame_bgr):
    """Box can currently be yellow OR green - find whichever is present.
    Returns rows, state, and an optional horizontal track estimate."""
    yellow = rows_matching(frame_bgr, YELLOW_LOWER, YELLOW_UPPER)
    green = rows_matching(frame_bgr, GREEN_LOWER, GREEN_UPPER)

    if yellow and green:
        y_size = yellow[1] - yellow[0]
        g_size = green[1] - green[0]
        box, state, lower, upper = ((green, "green", GREEN_LOWER, GREEN_UPPER)
                                    if g_size >= y_size else
                                    (yellow, "yellow", YELLOW_LOWER, YELLOW_UPPER))
    elif green:
        box, state, lower, upper = green, "green", GREEN_LOWER, GREEN_UPPER
    elif yellow:
        box, state, lower, upper = yellow, "yellow", YELLOW_LOWER, YELLOW_UPPER
    else:
        return None, None, None

    # Reject single-row color flashes and merged same-color UI regions.
    if not 15 <= box[1] - box[0] + 1 <= 70:
        return None, None, None

    # Horizontal localization is helpful but must never invalidate a valid
    # target. Some renderings only show separated horizontal border rows,
    # which is why the stricter 2D component detector rejected every frame.
    mask = np.all((frame_bgr >= lower) & (frame_bgr <= upper), axis=2)
    top, bottom = box
    cols = np.nonzero(mask[top:bottom + 1].any(axis=0))[0]
    track_x = None
    if len(cols):
        runs = []
        left = right = cols[0]
        for col in cols[1:]:
            if col <= right + 3:
                right = col
            else:
                runs.append((left, right))
                left = right = col
        runs.append((left, right))
        realistic = [(a, b) for a, b in runs if 3 <= b - a + 1 <= 220]
        if realistic:
            track_x = max(realistic, key=lambda r: r[1] - r[0])
    return box, state, track_x


def save_snapshot(frame_bgr, path_prefix="slider_snapshot"):
    """Write the captured region to disk, plus diagnostic images showing
    exactly which pixels matched each color rule. This is the fastest way
    to see WHY a detector is picking up the wrong thing."""
    try:
        from PIL import Image
    except ImportError:
        print("Snapshot needs Pillow: pip install pillow")
        return

    rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB for saving
    Image.fromarray(rgb.astype(np.uint8)).save(f"{path_prefix}_region.png")

    # Per-row match counts tell us what's fooling the detector.
    for name, lower, upper in [
        ("white", WHITE_LOWER, WHITE_UPPER),
        ("yellow", YELLOW_LOWER, YELLOW_UPPER),
        ("green", GREEN_LOWER, GREEN_UPPER),
    ]:
        mask = np.all((frame_bgr >= lower) & (frame_bgr <= upper), axis=2)
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"{path_prefix}_{name}_mask.png")
        counts = mask.sum(axis=1)
        nz = np.nonzero(counts)[0]
        if len(nz):
            print(f"  {name}: rows {nz.min()}-{nz.max()}, "
                  f"max {counts.max()} px/row, total matching rows {len(nz)}")
        else:
            print(f"  {name}: no matches")

    print(f"Snapshot saved as {path_prefix}_region.png (+ mask images) "
          f"in the folder you ran the script from.")


def largest_filled_run(frame_bgr, lower, upper, min_fill_frac=0.25, min_pixels=3):
    """Find the largest contiguous vertical run of *substantially filled* rows.

    This is the key to distinguishing the solid white slider BLOCK from the
    thin white OUTLINE that runs the entire length of the track. The outline
    contributes only 1-3 white pixels per row; the block contributes many.
    So we threshold on how many matching pixels a row contains, not merely
    whether it contains any, then take the biggest contiguous group.
    """
    mask = np.all((frame_bgr >= lower) & (frame_bgr <= upper), axis=2)
    counts = mask.sum(axis=1)
    if counts.max() == 0:
        return None

    thresh = max(min_pixels, counts.max() * min_fill_frac)
    rows = np.nonzero(counts >= thresh)[0]
    if len(rows) == 0:
        return None

    runs = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r
        else:
            runs.append((start, prev))
            start = prev = r
    runs.append((start, prev))

    top, bottom = max(runs, key=lambda t: t[1] - t[0])
    return int(top), int(bottom)


def find_white_slider(frame_bgr, track_x, previous_center=None):
    """Locate a realistic solid slider block, rejecting tall outline regions."""
    if track_x is None:
        left, right = 0, frame_bgr.shape[1] - 1
    else:
        box_left, box_right = track_x
        margin = max(48, (box_right - box_left + 1) * 2)
        left = max(0, box_left - margin)
        right = min(frame_bgr.shape[1] - 1, box_right + margin)
    track = frame_bgr[:, left:right + 1]
    mask = np.all((track >= WHITE_LOWER) & (track <= WHITE_UPPER), axis=2)
    counts = mask.sum(axis=1)
    if counts.max() == 0:
        return None
    rows = np.nonzero(counts >= max(3, counts.max() * 0.25))[0]
    runs = []
    if len(rows):
        start = prev = rows[0]
        for row in rows[1:]:
            if row == prev + 1:
                prev = row
            else:
                runs.append((start, prev))
                start = prev = row
        runs.append((start, prev))

    # Actual blocks in the logs are normally 10-30 px tall.  Permit a little
    # margin, but never accept the 100-240 px false regions from the old detector.
    candidates = [(a, b) for a, b in runs if 5 <= b - a + 1 <= 55]
    if not candidates:
        return None
    if previous_center is not None:
        nearby = [run for run in candidates
                  if abs((run[0] + run[1]) / 2.0 - previous_center)
                  <= MAX_FRAME_JUMP_PX]
        if not nearby:
            # Treat impossible teleportation as a missed frame rather than
            # steering toward an unrelated white UI element.
            return None
        return min(nearby,
                   key=lambda run: abs((run[0] + run[1]) / 2.0
                                       - previous_center))
    return max(candidates, key=lambda r: float(counts[r[0]:r[1] + 1].sum()))


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

class BotState:
    def __init__(self):
        self.region = None
        self.running = False
        self.should_exit = False
        self.holding = False
        self.corner1 = None
        self.corner2 = None
        self.snapshot_requested = False
        self.overlay_enabled = True
        self.overlay_data = None
        self.overlay_lock = threading.Lock()


def overlay_loop(state: BotState):
    """Draw a click-through diagnostic overlay above the selected region."""
    try:
        import tkinter as tk
    except ImportError:
        print("Visual overlay unavailable: tkinter is not installed.")
        return

    # Pure black is explicitly made invisible with the native Windows color
    # key below. Do not use a partially opaque fallback: that was the large
    # black rectangle seen in v13.2.
    transparent = "#000000"
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=transparent)
    if sys.platform != "win32":
        try:
            root.wm_attributes("-transparentcolor", transparent)
        except tk.TclError:
            print("Overlay transparency is only fully supported on Windows.")
            root.destroy()
            return
    canvas = tk.Canvas(root, bg=transparent, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    click_through_applied = False

    def rounded_panel(x1, y1, x2, y2, radius=10):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1,
            x2, y1 + radius, x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2, x1, y2,
            x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        canvas.create_polygon(points, smooth=True, fill="#111827",
                              outline="#8b5cf6", width=1)

    def bracket_marker(left, right, top, bottom, color, label):
        arm = max(5, min(12, (right - left) // 3))
        for x, direction in ((left, 1), (right, -1)):
            canvas.create_line(x, top, x + direction * arm, top,
                               fill=color, width=3)
            canvas.create_line(x, bottom, x + direction * arm, bottom,
                               fill=color, width=3)
            canvas.create_line(x, top, x, top + arm, fill=color, width=3)
            canvas.create_line(x, bottom, x, bottom - arm,
                               fill=color, width=3)
        canvas.create_text(right + 6, (top + bottom) / 2,
                           anchor="w", text=label, fill=color,
                           font=("Segoe UI", 9, "bold"))

    def refresh():
        nonlocal click_through_applied
        if state.should_exit:
            root.destroy()
            return
        region = state.region
        first_corner = state.corner1
        if (not state.overlay_enabled
                or (region is None and first_corner is None)):
            root.withdraw()
            root.after(50, refresh)
            return

        marker_mode = region is None
        if marker_mode:
            width, height = 190, 70
            window_left = int(first_corner[0] - 24)
            window_top = int(first_corner[1] - 35)
        else:
            width, height = region["width"], region["height"]
            window_left, window_top = region["left"], region["top"]
        root.geometry(f"{width}x{height}{window_left:+d}{window_top:+d}")
        root.deiconify()
        if not click_through_applied and sys.platform == "win32":
            try:
                import ctypes
                root.update_idletasks()
                # winfo_id() can be Tk's child window. GA_ROOT resolves the
                # actual top-level HWND that owns hit-testing and layering.
                child_hwnd = root.winfo_id()
                hwnd = ctypes.windll.user32.GetAncestor(child_hwnd, 2)
                if not hwnd:
                    hwnd = child_hwnd
                get_style = ctypes.windll.user32.GetWindowLongW
                set_style = ctypes.windll.user32.SetWindowLongW
                exstyle = get_style(hwnd, -20)
                # WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW |
                # WS_EX_NOACTIVATE. The color key makes black pixels fully
                # invisible; TRANSPARENT passes all mouse input to Roblox.
                set_style(hwnd, -20,
                          exstyle | 0x20 | 0x80000 | 0x80 | 0x08000000)
                color_key = 0x00000000  # COLORREF for RGB(0, 0, 0)
                ok = ctypes.windll.user32.SetLayeredWindowAttributes(
                    hwnd, color_key, 0, 0x1)
                if not ok:
                    raise ctypes.WinError()
                click_through_applied = True
            except Exception as exc:
                print(f"Overlay click-through warning: {exc}")

        with state.overlay_lock:
            data = dict(state.overlay_data) if state.overlay_data else None
        canvas.delete("all")
        if marker_mode:
            # The first selected screen coordinate is exactly (24, 35) in
            # this temporary window. Keep it visible until the second F9.
            x, y = 24, 35
            canvas.create_oval(x - 8, y - 8, x + 8, y + 8,
                               outline="#22d3ee", width=3)
            canvas.create_line(x - 15, y, x + 15, y,
                               fill="#22d3ee", width=2)
            canvas.create_line(x, y - 15, x, y + 15,
                               fill="#22d3ee", width=2)
            rounded_panel(48, 12, 182, 58, 9)
            canvas.create_text(59, 19, anchor="nw", text="POINT 1 SET",
                               fill="#e9d5ff",
                               font=("Segoe UI", 10, "bold"))
            canvas.create_text(59, 37, anchor="nw", text="Move, then F9",
                               fill="#94a3b8", font=("Segoe UI", 9))
            root.after(33, refresh)
            return

        # A subtle dashed selection boundary replaces the thick magenta box.
        canvas.create_rectangle(1, 1, width - 2, height - 2,
                                outline="#8b5cf6", width=1, dash=(5, 5))
        if data:
            track = data.get("track_x")
            left, right = track if track else (3, width - 4)
            left, right = max(4, left), min(width - 48, right)
            if right <= left:
                left, right = 4, max(12, width - 48)
            box = data.get("box")
            slider = data.get("slider")
            if box:
                bracket_marker(left, right, box[0], box[1],
                               "#c084fc", "TARGET")
            if slider:
                slider_mid = (slider[0] + slider[1]) / 2
                canvas.create_line(left + 3, slider_mid, right - 3, slider_mid,
                                   fill="#22d3ee", width=3)
                canvas.create_oval(left - 4, slider_mid - 4,
                                   left + 4, slider_mid + 4,
                                   fill="#22d3ee", outline="")
                canvas.create_text(right + 6, slider_mid, anchor="w",
                                   text="SLIDER", fill="#22d3ee",
                                   font=("Segoe UI", 9, "bold"))
            predicted = data.get("predicted_center")
            if predicted is not None:
                canvas.create_line(left, predicted, right, predicted,
                                   fill="#f59e0b", width=2, dash=(6, 4))
                canvas.create_polygon(left - 1, predicted,
                                      left + 8, predicted - 5,
                                      left + 8, predicted + 5,
                                      fill="#f59e0b", outline="")
            predicted_path = data.get("predicted_path") or []
            if predicted_path:
                path_x = left + max(4, (right - left) * 0.55)
                points = []
                for index, path_y in enumerate(predicted_path):
                    x = path_x + index * 5
                    points.extend((x, path_y))
                    radius = max(2, 5 - index * 0.6)
                    canvas.create_oval(x - radius, path_y - radius,
                                       x + radius, path_y + radius,
                                       fill="#fb7185", outline="#fecdd3",
                                       width=1)
                if len(points) >= 4:
                    canvas.create_line(*points, fill="#fb7185", width=2,
                                       dash=(3, 3))
                canvas.create_text(points[-2] + 7, points[-1], anchor="w",
                                   text="NEXT", fill="#fb7185",
                                   font=("Segoe UI", 8, "bold"))
            lines = data.get("lines", [])
            panel_width = min(width - 12, 500)
            panel_height = min(height - 12, 28 + len(lines) * 17)
            if panel_width > 100 and panel_height > 30:
                rounded_panel(6, 6, panel_width, panel_height, 11)
            for index, line in enumerate(lines):
                y = 13 + index * 17
                # Lavender/gray stay outside the detector's white threshold.
                color = "#e9d5ff" if index == 0 else "#cbd5e1"
                canvas.create_text(16, y, anchor="nw", text=line,
                                   fill=color, font=("Consolas", 9,
                                                    "bold" if index == 0 else "normal"))
        root.after(33, refresh)

    root.after(0, refresh)
    root.mainloop()


def set_hold(state: BotState, hold: bool):
    if hold and not state.holding:
        mouse.press(Button.left)
        state.holding = True
    elif not hold and state.holding:
        mouse.release(Button.left)
        state.holding = False


def set_pre_click_key(pressed: bool):
    """Send T as a hardware scan code so Roblox/DirectInput accepts it."""
    if sys.platform == "win32" and PRE_CLICK_KEY.lower() == "t":
        try:
            import ctypes

            ulong_ptr = (ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8
                         else ctypes.c_ulong)

            class KeyInput(ctypes.Structure):
                _fields_ = [
                    ("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ulong_ptr),
                ]

            class InputUnion(ctypes.Union):
                _fields_ = [("ki", KeyInput)]

            class Input(ctypes.Structure):
                _anonymous_ = ("union",)
                _fields_ = [("type", ctypes.c_ulong),
                            ("union", InputUnion)]

            flags = 0x0008 | (0x0002 if not pressed else 0)
            event = Input(type=1,
                          ki=KeyInput(0, 0x14, flags, 0, 0))
            sent = ctypes.windll.user32.SendInput(
                1, ctypes.byref(event), ctypes.sizeof(Input))
            if sent != 1:
                raise ctypes.WinError()
            return
        except Exception as exc:
            print(f"Hardware T input warning ({exc}); using fallback input.")
    key = pynput_keyboard.KeyCode.from_char(PRE_CLICK_KEY)
    (keyboard_ctrl.press if pressed else keyboard_ctrl.release)(key)


def click_once(state: BotState):
    """Hold PRE_CLICK_KEY for PRE_CLICK_HOLD_S, then send one left click.

    Any in-progress mouse hold is released first, otherwise the press/release
    pair would be swallowed by the existing held state and no click would
    land. The key release is in a finally block so an error can't leave the
    key stuck down.
    """
    set_hold(state, False)

    print(f"Idle interaction: holding '{PRE_CLICK_KEY.upper()}' for "
          f"{PRE_CLICK_HOLD_S:.1f}s")
    set_pre_click_key(True)
    try:
        time.sleep(PRE_CLICK_HOLD_S)
    finally:
        set_pre_click_key(False)

    # The interaction must finish and T must be released before the click.
    time.sleep(PRE_CLICK_RELEASE_DELAY_S)
    print("Idle interaction: T released, clicking")
    try:
        mouse.press(Button.left)
        time.sleep(0.03)
        mouse.release(Button.left)
        time.sleep(POST_CLICK_T_HOLD_S)
    finally:
        try:
            mouse.release(Button.left)
        except Exception:
            pass


def hard_click_once(state: BotState):
    """One raw fallback click, without the T-key sequence."""
    set_hold(state, False)
    mouse.press(Button.left)
    time.sleep(0.03)
    mouse.release(Button.left)


def bot_loop(state: BotState, debug=False, record_path=None, ml_model=None):
    MAX_MISSED_FRAMES = 6  # consecutive bad frames before we give up and release

    missed_frames = 0
    prev_center = None
    prev_time = None
    velocity = 0.0  # px/sec; positive = moving DOWN the screen
    prev_box_center = None
    locked_track_center = None
    locked_track_width = None
    target_velocity = 0.0
    target_acceleration = 0.0
    recent_accel = 0.0
    last_learning_action = None
    stable_action_frames = 0
    online_update_text = "waiting for stable motion"
    valid_detection_streak = 0
    up_accel_estimate = INITIAL_UP_ACCEL_PX_S2
    down_accel_estimate = INITIAL_DOWN_ACCEL_PX_S2
    has_detected_once = False   # gate: never idle-click before first detection
    last_detection_time = None
    idle_clicked = False        # whether we've already clicked this idle spell
    hard_idle_clicked = False   # one additional 20-second fallback click
    record_file = None
    recorder = None
    run_id = f"{int(time.time())}-{os.getpid()}"
    if record_path:
        existed = os.path.exists(record_path) and os.path.getsize(record_path) > 0
        record_file = open(record_path, "a", newline="", buffering=1)
        recorder = csv.writer(record_file)
        if not existed:
            recorder.writerow([
                "run_id", "time", "dt", "box_top", "box_bottom",
                "box_center", "slider_top", "slider_bottom", "slider_center",
                "slider_velocity", "target_velocity", "position_error",
                "holding", "box_state"
            ])

    capture_class = getattr(mss, "MSS", None)
    capture_context = capture_class() if capture_class else mss.mss()
    with capture_context as sct:
        while not state.should_exit:
            if not state.running or state.region is None:
                set_hold(state, False)
                missed_frames = 0
                valid_detection_streak = 0
                prev_center = None
                prev_box_center = None
                locked_track_center = None
                locked_track_width = None
                prev_time = None
                velocity = 0.0
                target_velocity = 0.0
                target_acceleration = 0.0
                recent_accel = 0.0
                last_learning_action = None
                stable_action_frames = 0
                # Clear idle tracking - a pause shouldn't look like the
                # minigame ending, and resuming shouldn't fire a stale click.
                last_detection_time = None
                with state.overlay_lock:
                    state.overlay_data = {
                        "lines": ["PAUSED", "F7 start/stop | F11 hide overlay"]
                    }
                # Pausing must not clear the post-click lock. Only a genuine
                # detection below is allowed to re-arm the end-of-round click.
                time.sleep(0.05)
                continue

            frame = grab(sct, state.region)

            # Snapshot request: dump exactly what the bot is seeing, so the
            # captured region can be inspected for false-positive sources.
            if state.snapshot_requested:
                state.snapshot_requested = False
                try:
                    save_snapshot(frame)
                except Exception as e:
                    print(f"Snapshot failed: {e}")

            box, box_state, track_x = find_box(frame)

            # Once the real meter is seen, lock onto its horizontal position.
            # When the minigame disappears, green scenery or yellow UI inside
            # a wide capture region must not masquerade as a live target and
            # continually re-arm the idle timer.
            rejected_track_candidate = None
            if box is not None and track_x is not None:
                candidate_width = track_x[1] - track_x[0] + 1
                candidate_center = (track_x[0] + track_x[1]) / 2.0
                if candidate_width > MAX_TARGET_WIDTH_PX:
                    rejected_track_candidate = (
                        f"target width {candidate_width:.0f}px")
                elif (locked_track_center is not None
                      and abs(candidate_center - locked_track_center)
                      > MAX_TRACK_X_DRIFT_PX):
                    rejected_track_candidate = (
                        f"track-x jump {candidate_center - locked_track_center:+.0f}px")
                elif (locked_track_width is not None
                      and abs(candidate_width - locked_track_width) > 18.0):
                    rejected_track_candidate = (
                        f"target width changed {candidate_width - locked_track_width:+.0f}px")
                if rejected_track_candidate:
                    box = None
                    box_state = None
                    track_x = None
            elif box is not None and locked_track_center is not None:
                rejected_track_candidate = "target has no horizontal track"
                box = None
                box_state = None

            slider = find_white_slider(frame, track_x, prev_center)

            if box is None or slider is None:
                missed_frames = min(missed_frames + 1, MAX_MISSED_FRAMES)
                idle_seconds = (time.perf_counter() - last_detection_time
                                if last_detection_time is not None else 0.0)
                # Color can flicker for one frame exactly when yellow changes
                # to green. Preserve motion across that one-frame gap; reset
                # only if the loss persists. The dt/jump checks below still
                # reject an implausible returning measurement.
                if missed_frames >= 2:
                    valid_detection_streak = 0
                    prev_center = None
                    prev_box_center = None
                    prev_time = None
                    velocity = 0.0
                    target_velocity = 0.0
                    target_acceleration = 0.0
                if debug:
                    status = "releasing" if missed_frames >= MAX_MISSED_FRAMES else "holding last state"
                    print(f"box={box} box_state={box_state} slider={slider} "
                          f"-> missed {missed_frames}/{MAX_MISSED_FRAMES}, {status}")
                with state.overlay_lock:
                    state.overlay_data = {
                        "box": box,
                        "slider": slider,
                        "track_x": track_x,
                        "lines": [
                            f"DETECTION MISSED {missed_frames}/{MAX_MISSED_FRAMES}",
                            f"box={box} slider={slider}",
                            (f"rejected={rejected_track_candidate}"
                             if rejected_track_candidate else
                             "waiting for valid meter geometry"),
                            f"idle={idle_seconds:.1f}s | T starts at "
                            f"{IDLE_CLICK_AFTER_S:.1f}s",
                            "F10 saves detector masks",
                        ],
                    }
                if missed_frames >= MAX_MISSED_FRAMES:
                    set_hold(state, False)
                    # Drop stale motion state; predicting from an outdated
                    # velocity after a gap produces a bad first decision.
                    prev_center = None
                    prev_time = None
                    velocity = 0.0
                    prev_box_center = None
                    target_velocity = 0.0
                    recent_accel = 0.0
                    last_learning_action = None
                    stable_action_frames = 0

                    # Idle click: the minigame has presumably ended. Only
                    # ever fires after we've successfully detected it at
                    # least once, so the bot won't click blindly at startup
                    # or if the region was set up wrong.
                    if (has_detected_once and last_detection_time is not None
                            and not idle_clicked):
                        idle_for = time.perf_counter() - last_detection_time
                        if idle_for >= IDLE_CLICK_AFTER_S:
                            if debug:
                                print(f"  idle {idle_for:.1f}s -> hold "
                                      f"'{PRE_CLICK_KEY}' {PRE_CLICK_HOLD_S}s "
                                      f"then click (waiting for next detection)")
                            click_once(state)
                            # Latch it. Only a real detection clears this,
                            # so no further clicks happen until the minigame
                            # is actually seen again.
                            idle_clicked = True

                    # Explicit fallback requested by the user: one raw click
                    # at 20 seconds even though the normal idle click already
                    # fired. It still cannot repeat again before detection.
                    if (has_detected_once and last_detection_time is not None
                            and not hard_idle_clicked):
                        idle_for = time.perf_counter() - last_detection_time
                        if idle_for >= HARD_CLICK_AFTER_S:
                            if debug:
                                print(f"  idle {idle_for:.1f}s -> HARD fallback click "
                                      "(locked until detection)")
                            hard_click_once(state)
                            hard_idle_clicked = True
                continue

            missed_frames = 0
            valid_detection_streak += 1
            if not has_detected_once:
                has_detected_once = True
                if debug:
                    print("  first detection - idle click now armed")
            last_detection_time = time.perf_counter()
            idle_clicked = False  # genuine box + slider detection re-arms click
            hard_idle_clicked = False

            box_top, box_bottom = box
            slider_top, slider_bottom = slider
            box_center = (box_top + box_bottom) / 2
            slider_center = (slider_top + slider_bottom) / 2
            if locked_track_center is None and track_x is not None:
                locked_track_center = (track_x[0] + track_x[1]) / 2.0
                locked_track_width = track_x[1] - track_x[0] + 1
                if debug:
                    print(f"  locked meter track at x={locked_track_center:.1f} "
                          f"width={locked_track_width:.0f}px")

            active_learning_action = bool(state.holding)
            if active_learning_action == last_learning_action:
                stable_action_frames += 1
            else:
                last_learning_action = active_learning_action
                stable_action_frames = 1
            online_update_text = f"learning gate {stable_action_frames}/3"

            # --- Velocity estimate: the anti-overshoot term ---
            now = time.perf_counter()
            frame_dt = 0.0
            motion_sample_valid = False
            reversal = False
            if prev_center is not None and prev_time is not None:
                dt = now - prev_time
                frame_dt = dt
                slider_jump = abs(slider_center - prev_center)
                box_jump = (abs(box_center - prev_box_center)
                            if prev_box_center is not None else 0.0)
                if 0.004 <= dt <= MAX_VALID_FRAME_DT_S \
                        and slider_jump <= MAX_FRAME_JUMP_PX \
                        and box_jump <= MAX_FRAME_JUMP_PX:
                    motion_sample_valid = True
                    old_velocity = velocity
                    raw_velocity = (slider_center - prev_center) / dt  # px/sec
                    raw_velocity = max(-MAX_TRACKED_SPEED_PX_S,
                                       min(MAX_TRACKED_SPEED_PX_S, raw_velocity))

                    # Stall detection: if the slider has essentially stopped
                    # moving (pinned against the top or bottom of the track),
                    # collapse the estimate immediately instead of letting
                    # smoothing carry a stale high velocity for several
                    # frames. That stale value is what makes behavior go
                    # wrong right at the track limits.
                    if abs(raw_velocity) < STALL_VELOCITY_PX_S:
                        velocity = raw_velocity
                    else:
                        velocity = (VELOCITY_SMOOTHING * velocity
                                    + (1.0 - VELOCITY_SMOOTHING) * raw_velocity)

                    measured_accel = (velocity - old_velocity) / dt
                    if ml_model is not None:
                        learned, residual = ml_model.observe(
                            old_velocity, active_learning_action,
                            recent_accel, dt, active_learning_action,
                            measured_accel, stable_action_frames)
                        if learned:
                            online_update_text = (
                                f"learned {'HOLD' if active_learning_action else 'RELEASE'} "
                                f"residual={residual:+.0f}")
                            try:
                                if ml_model.save():
                                    online_update_text += " | autosaved"
                            except Exception as exc:
                                online_update_text = f"autosave failed: {exc}"
                    recent_accel = (0.65 * recent_accel
                                    + 0.35 * max(-5000.0,
                                                 min(5000.0, measured_accel)))
                    if state.holding and measured_accel < -100.0:
                        sample = min(MAX_LEARNED_ACCEL_PX_S2,
                                     max(MIN_LEARNED_ACCEL_PX_S2,
                                         -measured_accel))
                        up_accel_estimate = ((1.0 - ACCEL_LEARNING_RATE)
                                             * up_accel_estimate
                                             + ACCEL_LEARNING_RATE * sample)
                    elif not state.holding and measured_accel > 100.0:
                        sample = min(MAX_LEARNED_ACCEL_PX_S2,
                                     max(MIN_LEARNED_ACCEL_PX_S2,
                                         measured_accel))
                        down_accel_estimate = ((1.0 - ACCEL_LEARNING_RATE)
                                               * down_accel_estimate
                                               + ACCEL_LEARNING_RATE * sample)
                    if prev_box_center is not None:
                        old_target_velocity = target_velocity
                        raw_target_velocity = (box_center - prev_box_center) / dt
                        raw_target_velocity = max(-MAX_TARGET_SPEED_PX_S,
                                                  min(MAX_TARGET_SPEED_PX_S,
                                                      raw_target_velocity))
                        reversal = (abs(raw_target_velocity)
                                    >= TARGET_REVERSAL_SPEED_PX_S
                                    and abs(target_velocity)
                                    >= TARGET_REVERSAL_SPEED_PX_S
                                    and raw_target_velocity * target_velocity < 0)
                        smoothing = (0.25 if reversal
                                     else TARGET_VELOCITY_SMOOTHING)
                        target_velocity = (smoothing * target_velocity
                                           + (1.0 - smoothing)
                                           * raw_target_velocity)
                        measured_target_accel = ((target_velocity
                                                  - old_target_velocity) / dt)
                        measured_target_accel = max(
                            -MAX_TARGET_ACCEL_PX_S2,
                            min(MAX_TARGET_ACCEL_PX_S2,
                                measured_target_accel))
                        accel_smoothing = (0.25 if reversal
                                           else TARGET_ACCELERATION_SMOOTHING)
                        target_acceleration = (
                            accel_smoothing * target_acceleration
                            + (1.0 - accel_smoothing)
                            * measured_target_accel)
                else:
                    # A detector jump or timing gap is not physical motion.
                    velocity = 0.0
                    target_velocity = 0.0
                    target_acceleration = 0.0
                    recent_accel = 0.0
                    last_learning_action = None
                    stable_action_frames = 0
            prev_center = slider_center
            prev_box_center = box_center
            prev_time = now

            # Steer on where the slider is HEADING, not where it sits now.
            # Falling (positive velocity, moving down) gets a longer horizon
            # because gravity keeps accelerating it; rising is under
            # controlled thrust and needs less anticipation.
            # Predict relative stopping position. A fixed lookahead cannot work
            # equally well at 30 px/s and 250 px/s; stopping distance grows with
            # speed squared. The acceleration estimates update from observed
            # hold/release response throughout the run.
            relative_velocity = velocity - target_velocity
            # One valid pair is sufficient: the current frame and the prior
            # detected frame. A fixed three-frame streak kept slow/flickering
            # captures permanently in "warming up" despite valid motion.
            velocity_ready = (motion_sample_valid
                              and valid_detection_streak >= VELOCITY_WARMUP_FRAMES)
            if not velocity_ready:
                relative_velocity = 0.0
            if relative_velocity > 0:
                stopping_distance = (relative_velocity * relative_velocity
                                     / (2.0 * up_accel_estimate))
            else:
                stopping_distance = -(relative_velocity * relative_velocity
                                      / (2.0 * down_accel_estimate))
            stopping_distance = max(-MAX_STOPPING_DISTANCE_PX,
                                    min(MAX_STOPPING_DISTANCE_PX,
                                        stopping_distance))
            effective_latency = max(CONTROL_LATENCY_S,
                                    min(0.10, frame_dt * 0.70))
            predicted_error = (slider_center - box_center
                               + relative_velocity * effective_latency
                               - 0.5 * target_acceleration
                               * effective_latency * effective_latency
                               + stopping_distance)
            predicted_center = box_center + predicted_error

            # Clamp the prediction to the actual track. Near the top/bottom
            # the slider physically cannot keep going, but a stale velocity
            # estimate would project it off the end of the track and produce
            # a nonsense decision. The slider also clamps hard at the limits,
            # which makes the raw velocity estimate briefly meaningless.
            frame_h = frame.shape[0]
            predicted_center = max(0.0, min(float(frame_h - 1), predicted_center))

            overlapping = (slider_top <= box_bottom) and (slider_bottom >= box_top)

            if debug:
                direction = "down" if velocity > 0 else "up"
                velocity_text = (f"{velocity:+.0f}({direction})"
                                 if velocity_ready else "warming")
                print(f"box=[{box_top},{box_bottom}]({box_state}) "
                      f"slider=[{slider_top},{slider_bottom}] "
                      f"v={velocity_text} stop={stopping_distance:+.0f} "
                      f"tv={target_velocity:+.0f} ta={target_acceleration:+.0f} "
                      f"rel={relative_velocity:+.0f} "
                      f"dt={frame_dt * 1000:.0f}ms "
                      f"vready={velocity_ready} "
                      f"a={up_accel_estimate:.0f}/{down_accel_estimate:.0f} "
                      f"err={predicted_error:+.0f} target={box_center:.0f} "
                      f"hold={state.holding}")

            currently_good = (box_state == "green") or overlapping

            # Holding pushes the slider UP = toward SMALLER row indices.
            ml_cost = None
            ml_path_errors = []
            edge_override = None
            if ml_model is not None and velocity_ready:
                box_half = (box_bottom - box_top + 1) / 2.0
                slider_half = (slider_bottom - slider_top + 1) / 2.0
                allowed_error = max(1.0, box_half - slider_half
                                    - EDGE_SAFETY_MARGIN_PX)
                desired_hold, ml_cost, ml_path_errors = ml_model.choose_action(
                    slider_center - box_center, velocity, target_velocity,
                    target_acceleration,
                    state.holding, allowed_error, recent_accel, frame_dt)

                # Hard edge-aware safety layer. Convert the predicted center
                # into predicted slider edges and keep both inside an inset
                # target interval. This catches boundary escapes even if the
                # learned model selects a poor action.
                predicted_slider_top = predicted_center - slider_half
                predicted_slider_bottom = predicted_center + slider_half
                safe_top = box_top + EDGE_SAFETY_MARGIN_PX
                safe_bottom = box_bottom - EDGE_SAFETY_MARGIN_PX
                # Intervene only while moving toward the threatened edge.
                # The previous unconditional override fought the ML controller
                # and caused extra oscillation when momentum was already safe.
                if (predicted_slider_bottom > safe_bottom
                        and relative_velocity > EDGE_OVERRIDE_MIN_SPEED_PX_S):
                    desired_hold = True       # push upward from lower edge
                    edge_override = "LOWER"
                elif (predicted_slider_top < safe_top
                      and relative_velocity < -EDGE_OVERRIDE_MIN_SPEED_PX_S):
                    desired_hold = False      # fall away from upper edge
                    edge_override = "UPPER"
                set_hold(state, desired_hold)
            else:
                if currently_good:
                    if predicted_error > DEAD_ZONE_PX:
                        set_hold(state, True)
                    elif predicted_error < -DEAD_ZONE_PX:
                        set_hold(state, False)
                else:
                    if predicted_error > 0:
                        set_hold(state, True)
                    else:
                        set_hold(state, False)

            raw_error = slider_center - box_center
            if ml_model is None:
                ml_status = "ML unavailable: model did not load"
            elif not velocity_ready:
                ml_status = (f"ML WAITING: valid detections "
                             f"{valid_detection_streak}/{VELOCITY_WARMUP_FRAMES}, "
                             f"dt={frame_dt * 1000:.0f}ms")
            else:
                ml_status = f"ML active cost={ml_cost:.1f}"
            with state.overlay_lock:
                state.overlay_data = {
                    "box": box,
                    "slider": slider,
                    "track_x": track_x,
                    "predicted_center": predicted_center,
                    "predicted_path": [box_center + item
                                       for item in ml_path_errors],
                    "lines": [
                        f"BOX {box_top}-{box_bottom} {box_state.upper()}  "
                        f"SLIDER {slider_top}-{slider_bottom}",
                        f"error={raw_error:+.1f}px predicted={predicted_error:+.1f}px",
                        f"slider_v={velocity:+.1f} target_v={target_velocity:+.1f} "
                        f"relative={relative_velocity:+.1f}",
                        f"target_a={target_acceleration:+.0f}px/s² "
                        f"pattern={'REVERSAL' if reversal else 'tracking'}",
                        f"stop={stopping_distance:+.1f}px dt={frame_dt * 1000:.0f}ms",
                        f"decision={'HOLD' if state.holding else 'RELEASE'} "
                        f"edge={edge_override or 'safe'}",
                        ml_status,
                        (f"ONLINE {online_update_text} | samples "
                         f"R={ml_model.sample_counts[0]} "
                         f"H={ml_model.sample_counts[1]}"
                         if ml_model is not None else "ONLINE model unavailable"),
                    ],
                }

            if recorder is not None:
                recorder.writerow([
                    run_id, f"{now:.9f}", f"{frame_dt:.6f}",
                    box_top, box_bottom, f"{box_center:.3f}",
                    slider_top, slider_bottom, f"{slider_center:.3f}",
                    f"{velocity:.6f}", f"{target_velocity:.6f}",
                    f"{slider_center - box_center:.6f}",
                    int(state.holding), box_state
                ])

            if debug and ml_model is not None and ml_cost is not None:
                print(f"  ML action={'HOLD' if state.holding else 'RELEASE'} "
                      f"cost={ml_cost:.1f} edge={edge_override or 'safe'} "
                      f"gaps={slider_top - box_top:+.0f}/"
                      f"{box_bottom - slider_bottom:+.0f}")

            # No sleep here - the loop runs as fast as the screen grab
            # allows, so hold/release reacts with minimum latency.

    set_hold(state, False)
    if ml_model is not None:
        try:
            if ml_model.save(force=True):
                print(f"Saved online model: {ml_model.path}")
        except Exception as exc:
            print(f"Could not save online model: {exc}")
    if record_file is not None:
        record_file.close()


# ---------------------------------------------------------------------------
# Hotkeys
# ---------------------------------------------------------------------------

def main():
    debug = "--debug" in sys.argv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    record_path = None
    if "--record" in sys.argv:
        index = sys.argv.index("--record")
        if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--"):
            record_path = os.path.abspath(sys.argv[index + 1])
        else:
            record_path = os.path.join(script_dir, "slider_training_data.csv")

    online_model_path = os.path.join(script_dir,
                                     "slider_model_v5_online.npz")
    v4_models = glob.glob(os.path.join(script_dir, "slider_model_v4*.npz"))
    v4_reference_path = (max(v4_models, key=os.path.getmtime)
                         if v4_models else None)
    model_path = online_model_path
    seed_online_model = False
    if "--model" in sys.argv:
        index = sys.argv.index("--model")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--model requires a .npz path")
        model_path = os.path.abspath(sys.argv[index + 1])
    elif not os.path.exists(model_path):
        # Browsers may rename repeated downloads to slider_model_v5_online(1).npz.
        # Select the newest matching model so it is still loaded automatically.
        downloaded_models = glob.glob(os.path.join(script_dir,
                                                   "slider_model_v5_online*.npz"))
        if downloaded_models:
            model_path = max(downloaded_models, key=os.path.getmtime)
        else:
            # First online run: seed safely from the existing robust v4 model
            # and immediately save a separate v5 file. The v4 file is never
            # modified, so rollback remains possible.
            if v4_reference_path:
                model_path = v4_reference_path
                seed_online_model = True
    ml_model = None
    if os.path.exists(model_path):
        try:
            ml_model = LearnedDynamicsModel(model_path)
            if seed_online_model:
                ml_model.path = online_model_path
                ml_model.save(force=True)
                model_path = online_model_path
                print(f"Created online model from v4 seed: {model_path}")
            if v4_reference_path:
                try:
                    correction = ml_model.set_safety_reference(v4_reference_path)
                    ml_model.save(force=True)
                    print(f"Online safety reference: {v4_reference_path}")
                    if correction > 0.01:
                        print(f"  repaired excessive online drift by up to "
                              f"{correction:.1f} coefficient units")
                except Exception as guard_exc:
                    print(f"Could not apply v4 safety reference: {guard_exc}")
            else:
                print("Warning: no slider_model_v4*.npz safety reference found; "
                      "online learning will be anchored to the loaded model.")
        except Exception as exc:
            print(f"Could not load ML model '{model_path}': {exc}")

    state = BotState()

    overlay_thread = threading.Thread(target=overlay_loop,
                                      args=(state,), daemon=True)
    overlay_thread.start()

    bot_thread = threading.Thread(target=bot_loop,
                                  args=(state, debug, record_path, ml_model),
                                  daemon=True)
    bot_thread.start()

    print(f"Slider Auto-Hold Bot {BOT_VERSION} ready.")
    if ml_model is not None:
        print(f"  prediction: ML model-predictive control ({model_path})")
        print(f"  online learning: enabled; autosaves every "
              f"{ONLINE_SAVE_EVERY} accepted updates")
        print(f"  learned samples: release={ml_model.sample_counts[0]}, "
              f"hold={ml_model.sample_counts[1]}")
    else:
        print("  prediction: adaptive fallback "
              "(no slider_model_v5_online.npz yet)")
    if record_path:
        print(f"  recording training data: {record_path}")
    print("  F9  = set region corner (press once at each opposite corner)")
    print("  F7  = start/stop")
    print("  F10 = save snapshot of what the bot currently sees (diagnostic)")
    print("  F11 = show/hide live visual overlay")
    print("  F8  = panic stop & exit")
    print("\nTo set the region: hover your mouse at one corner of the slider")
    print("track and tap F9, then hover the opposite corner and tap F9 again.")
    print("Keep the region TIGHT around just the track - extra screen area")
    print("is what causes false color matches.\n")

    def on_f10():
        if state.region is None:
            print("Set a region first with F9 (twice).")
            return
        print("Taking snapshot...")
        state.snapshot_requested = True

    def on_f9():
        if state.corner1 is None:
            capture_corner(state, "first")
        else:
            capture_corner(state, "second")

    def on_f7():
        if state.region is None:
            print("Set a region first with F9 (twice).")
            return
        state.running = not state.running
        print("Running" if state.running else "Paused")

    def on_f11():
        state.overlay_enabled = not state.overlay_enabled
        print("Overlay visible" if state.overlay_enabled else "Overlay hidden")

    def on_f8():
        print("Panic stop.")
        state.running = False
        state.should_exit = True
        set_hold(state, False)
        # In case F8 lands during the pre-click key hold.
        try:
            set_pre_click_key(False)
        except Exception:
            pass
        listener.stop()

    with pynput_keyboard.GlobalHotKeys({
        "<f9>": on_f9,
        "<f7>": on_f7,
        "<f8>": on_f8,
        "<f10>": on_f10,
        "<f11>": on_f11,
    }) as listener:
        listener.join()

    print("Exited cleanly.")


if __name__ == "__main__":
    main()
