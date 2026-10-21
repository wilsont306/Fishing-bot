# Slider Auto-Hold Bot

A Windows computer-vision controller for a vertical slider minigame. It detects the slider and moving target, predicts their motion, and decides when to hold or release the mouse. The current build includes a live diagnostic overlay, guarded online learning, target-motion prediction, automatic idle interactions, and a robust offline training path.

> Use automation responsibly and follow the rules of the game or service where you run it.

## Features

- Color-based slider and target detection inside a user-selected region
- Predictive hold/release control with separate upward and downward dynamics
- Guarded online learning that persists between sessions
- Stable v4 model retained as a safety reference and rollback point
- Target velocity, acceleration, and reversal prediction
- Click-through live overlay showing detections and predicted positions
- Diagnostic screenshots and optional CSV recording
- Idle recovery: holds `T`, releases it, then clicks after detection is lost
- 20-second hard fallback click if nothing has been detected

## Requirements

- Windows
- Python 3.10 or newer
- A tightly selected capture region around the slider track

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Keep `slider_bot.py`, `slider_model_v4.npz`, and `slider_model_v5_online.npz` in the same folder, then run:

```powershell
python slider_bot.py
```

For detailed controller output:

```powershell
python slider_bot.py --debug
```

To also record a training CSV:

```powershell
python slider_bot.py --record slider_training_data.csv
```

## Controls

| Key | Action |
| --- | --- |
| `F9` | Mark one capture-region corner; press again at the opposite corner |
| `F7` | Start or pause the controller |
| `F8` | Panic stop, release inputs, and exit |
| `F10` | Save a diagnostic snapshot of the selected region |
| `F11` | Show or hide the click-through live overlay |

Select the region tightly around the slider track. Extra background pixels can create false color matches and reduce accuracy.

## Learning behavior

`slider_model_v5_online.npz` is the active model. The bot updates it only from accepted, high-confidence transitions and autosaves periodically, so learning carries across launches.

`slider_model_v4.npz` is the conservative safety reference. At startup, the bot uses it to limit excessive drift in the online model. Keep both files beside the script.

If you want a clean online-learning reset, back up or rename the current v5 file. On the next launch, the bot will seed a fresh v5 model from the v4 reference.

## Offline training

Record several minutes of representative gameplay, then train a fresh robust reference model:

```powershell
python train_slider_model.py slider_training_data.csv slider_model_v4.npz
```

The trainer quarantines weak sessions, balances samples across recording runs, trims detector outliers, and fits separate hold and release dynamics. Back up your existing model before replacing it.

## Repository files

| File | Purpose |
| --- | --- |
| `slider_bot.py` | Main controller, detector, overlay, and online learner |
| `slider_model_v5_online.npz` | Persisted online model |
| `slider_model_v4.npz` | Stable safety-reference model |
| `train_slider_model.py` | Robust offline trainer for recorded CSV data |
| `requirements.txt` | Python dependencies |

## Repository description

Computer-vision slider controller with live diagnostics, guarded online learning, target-motion prediction, and automatic idle recovery.
