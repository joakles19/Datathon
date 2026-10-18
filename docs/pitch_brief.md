# HearHere: Pitch Brief for PowerPoint Generation

## One-line pitch

HearHere turns important environmental sounds into configurable visual prompts, helping people who may miss audio cues notice events such as a door knock, siren, or possible glass break.

## Deck objective

Create a clear, human-centred product pitch for a working assistive prototype. The story should show the problem, the end-to-end concept, the evidence from the first model baseline, and the path to a safer real-world pilot.

## Audience and tone

- Audience: datathon judges, product and accessibility stakeholders, and technical reviewers.
- Tone: empathetic, practical, evidence-led, and honest about prototype limitations.
- Visual style: high contrast, accessible typography, simple alert colours, and a calm interface. Avoid fear-based imagery or claims that the system is an emergency warning device.

## The problem

Many everyday situations communicate through sound first: someone knocks at the door, a siren passes nearby, or glass may have broken. A person who cannot hear the sound clearly, is wearing headphones, or is in a noisy environment can miss the cue and lose context about what is happening around them.

The product opportunity is to make selected sound events visible and actionable without forcing the user to monitor a screen continuously.

## The solution

HearHere listens for a focused set of environmental sounds and presents a visual prompt only when the event matters to the user.

Core experience:

1. Capture a short audio clip.
2. Standardise the clip to mono, 16 kHz, and a five-second analysis window.
3. Extract 40 MFCC features: 20 coefficient means plus 20 population standard deviations.
4. Classify the clip with a Random Forest model across ten sound classes.
5. Apply per-sound preferences, confidence thresholds, priorities, cooldowns, snooze, and acknowledgement state.
6. Store an event history so the user can review what was detected.

The six alert-oriented classes are `door_wood_knock`, `siren`, `glass_breaking`, `clock_alarm`, `crying_baby`, and `car_horn`. Other classes provide context and help the classifier distinguish ordinary background sounds.

## What exists today

### Working model and evaluation flow

- Reproducible audio preprocessing and feature extraction in `hearhere_audio/`.
- Random Forest baseline with 300 trees and fixed random seed 42.
- ESC-50 selected classes: car horn, clock alarm, crying baby, door knock, engine, glass breaking, rain, siren, vacuum cleaner, and wind.
- Official fold split: folds 1-3 for training, fold 4 for validation, fold 5 reserved for final evaluation.
- Training data: 240 clips.
- Validation data: 80 clips.
- Model artifact: `models/hearhere_rf.joblib`.
- Measured report: `reports/validation_metrics.json`.

### Backend interaction scaffold

The FastAPI backend already exposes the product control loop:

- `POST /classify` for an audio clip.
- `GET /events` for recent detections.
- `POST /events/{event_id}/ack` for acknowledgement.
- `GET/PUT /prefs/{label}` for per-sound settings.
- `POST /snooze` for temporary suppression.

The trained model is loaded by the endpoint. It scores each WAV, applies the per-class alert thresholds, manages cooldown and snooze, and records events for acknowledgement and history. The browser client sends converted WAV windows to this endpoint and displays the returned action.

## Evidence from the current baseline

The following results are measured on ESC-50 Fold 4 only. They are a technical baseline, not a claim of real-world microphone or emergency performance.

| Metric | Result |
|---|---:|
| Validation accuracy | 77.5% |
| Macro F1 | 76.39% |
| Weighted F1 | 76.39% |
| Validation samples | 80 |
| Sound classes | 10 |

### Important sound performance

| Sound | Precision | Recall | F1 | Interpretation |
|---|---:|---:|---:|---|
| Door knock | 72.7% | 100.0% | 84.2% | Highest recall in this run |
| Glass breaking | 100.0% | 100.0% | 100.0% | Strong result on this validation split |
| Siren | 71.4% | 62.5% | 66.7% | Needs more varied examples and noise testing |
| Clock alarm | 87.5% | 87.5% | 87.5% | Good result on this validation split |
| Crying baby | 77.8% | 87.5% | 82.4% | Good catch rate, with room to improve precision |
| Car horn | 30.0% | 37.5% | 33.3% | Weakest alert class; thresholding has many false positives |

### What the confusion matrix suggests

- Rain was recognised correctly in all 8 validation examples.
- Door knock and glass breaking were each detected in 8 of 8 examples.
- Siren was detected in 5 of 8 examples.
- Wind was sometimes confused with siren, which matters for false-alert control.
- Engine was the weakest class in this run: 4 of 8 examples were correct.

Use these observations to motivate threshold tuning and more diverse recordings; do not present them as universal performance guarantees.

## Suggested slide narrative

### Slide 1 - HearHere

**Headline:** Make important sounds visible.

Show a simple visual journey from an unseen sound cue to a clear visual prompt. Keep the product name and the six priority events prominent: door knock, siren, glass break, clock alarm, crying baby, and car horn.

### Slide 2 - The missed cue problem

Show a relatable home or street scenario where an audio-only cue can be missed. Explain that the product focuses on awareness and context, not medical diagnosis or emergency dispatch.

### Slide 3 - The user experience

Show the flow: listen -> recognise -> decide -> prompt -> acknowledge. Use a mock alert timeline with event label, priority, timestamp, and acknowledgement state.

### Slide 4 - From audio to visual context

Show the technical pipeline: five-second audio window -> 40 MFCC features -> Random Forest -> policy layer -> visual alert and event history.

### Slide 5 - What the prototype recognises

Display the ten classes as a balanced sound map. Visually distinguish the six alert-oriented classes from ordinary/context classes.

### Slide 6 - Baseline evidence

Lead with 77.5% validation accuracy and 76.39% macro F1. Add a compact bar chart for precision, recall, and F1 of the six alert classes.

### Slide 7 - Why policy matters

Show that a classifier score alone is not an alert. Demonstrate configurable thresholds, priority, cooldown, snooze, acknowledgement, and event history. Mention wind/siren confusion as the reason false-alert controls are part of the product design.

### Slide 8 - Responsible prototype boundaries

State clearly: Fold 5 final evaluation is still pending, and real microphone and diverse-environment testing is still needed. The local frontend/backend/model flow is implemented, but it is not a validated emergency-warning device. This transparency is part of the product credibility.

### Slide 9 - Roadmap and ask

Show the next steps:

1. Test the full flow on additional recordings and supported browsers.
2. Deploy the API to the shared environment and verify it there.
3. Freeze the model and policy, then run the reserved Fold 5 evaluation.
4. Test with real microphones, varied rooms, background noise, and accessibility users.
5. Measure alert usefulness, acknowledgement time, and user trust.

Close with the ask: support a focused pilot to turn a measured audio baseline into a reliable, user-configurable awareness tool.

## Demo script for the presenter

1. Upload or simulate a short clip.
2. Show the predicted sound and confidence.
3. Show the policy decision: alert, ignore, or suppress because of snooze/cooldown.
4. Show the visual prompt for a priority sound.
5. Acknowledge the event and open the event history.
6. Change a preference or snooze a sound to demonstrate user control.

For a demo, label simulated results clearly. In live mode, show the model version and distinguish its top-scoring class from the backend-selected alert label when those differ.

## Claims the deck may make

- HearHere is a working assistive prototype with a reproducible audio classification baseline.
- The current model reaches 77.5% accuracy and 76.39% macro F1 on the measured ESC-50 Fold 4 validation split.
- The product combines sound recognition with a user-facing alert policy, rather than exposing raw classifier scores alone.
- The design prioritises user control through thresholds, cooldowns, snooze, acknowledgement, and history.

## Claims to avoid

- Do not call HearHere a validated emergency-warning system.
- Do not describe Random Forest scores as calibrated real-world probabilities.
- Do not claim final test performance before Fold 5 is run.
- Do not claim shared or LinuxONE deployment until those environments have been tested.
- Do not claim clinical, safety-critical, or universal accessibility outcomes without user testing.

## Source files for the PowerPoint agent

- Product and experiment scope: `README.md`
- Audio feature contract: `hearhere_audio/features.py` and `hearhere_audio/config.py`
- Training and evaluation flow: `scripts/train.py`
- Measured validation metrics: `reports/validation_metrics.json`
- Confusion matrix: `reports/confusion_matrix.csv`
- Model identity and data split: `models/model_metadata.json`
- Backend policy and API scaffold: `backends/main.py`
- Artifact and evaluation caveats: `docs/model-artifacts.md`

## Data and evaluation note

The selected ESC-50 data is used under the repository's recorded `CC BY-NC 3.0` note. The source dataset is not included in Git. The Fold 5 split must remain untouched until the model and alert policy are fixed; otherwise it is no longer an independent final evaluation.
