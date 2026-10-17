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
4. Classify the clip with a Random Forest model across eight sound classes.
5. Apply per-sound preferences, confidence thresholds, priorities, cooldowns, snooze, and acknowledgement state.
6. Store an event history so the user can review what was detected.

The three alert-oriented classes are `door_wood_knock`, `siren`, and `glass_breaking`. Other classes provide context and help the classifier distinguish ordinary background sounds.

## What exists today

### Working model and evaluation flow

- Reproducible audio preprocessing and feature extraction in `hearhere_audio/`.
- Random Forest baseline with 300 trees and fixed random seed 42.
- ESC-50 selected classes: door knock, engine, glass breaking, keyboard typing, rain, siren, vacuum cleaner, and wind.
- Official fold split: folds 1-3 for training, fold 4 for validation, fold 5 reserved for final evaluation.
- Training data: 192 clips.
- Validation data: 64 clips.
- Model artifact: `models/hearhere_rf.joblib`.
- Measured report: `reports/validation_metrics.json`.

### Backend interaction scaffold

The FastAPI backend already exposes the product control loop:

- `POST /classify` for an audio clip.
- `GET /events` for recent detections.
- `POST /events/{event_id}/ack` for acknowledgement.
- `GET/PUT /prefs/{label}` for per-sound settings.
- `POST /snooze` for temporary suppression.

The policy layer is implemented, including alert priority, cooldown, snooze, acknowledgement, and event logging. The current `predict()` function is still a placeholder returning a fixed siren score; wiring the trained model into this endpoint is a next implementation step.

## Evidence from the current baseline

The following results are measured on ESC-50 Fold 4 only. They are a technical baseline, not a claim of real-world microphone or emergency performance.

| Metric | Result |
|---|---:|
| Validation accuracy | 81.25% |
| Macro F1 | 80.93% |
| Weighted F1 | 80.93% |
| Validation samples | 64 |
| Sound classes | 8 |

### Important sound performance

| Sound | Precision | Recall | F1 | Interpretation |
|---|---:|---:|---:|---|
| Door knock | 100.0% | 87.5% | 93.3% | Strongest alert-oriented result in this run |
| Glass breaking | 63.6% | 87.5% | 73.7% | Good catch rate, but false positives need attention |
| Siren | 70.0% | 87.5% | 77.8% | Good catch rate, with room to improve precision |

### What the confusion matrix suggests

- Rain was recognised correctly in all 8 validation examples.
- Door knock and glass breaking were each detected in 7 of 8 examples.
- Siren was detected in 7 of 8 examples.
- Wind was sometimes confused with siren, which matters for false-alert control.
- Engine was the weakest class in this run: 4 of 8 examples were correct.

Use these observations to motivate threshold tuning and more diverse recordings; do not present them as universal performance guarantees.

## Suggested slide narrative

### Slide 1 - HearHere

**Headline:** Make important sounds visible.

Show a simple visual journey from an unseen sound cue to a clear visual prompt. Keep the product name and the three priority events prominent: door knock, siren, glass break.

### Slide 2 - The missed cue problem

Show a relatable home or street scenario where an audio-only cue can be missed. Explain that the product focuses on awareness and context, not medical diagnosis or emergency dispatch.

### Slide 3 - The user experience

Show the flow: listen -> recognise -> decide -> prompt -> acknowledge. Use a mock alert timeline with event label, priority, timestamp, and acknowledgement state.

### Slide 4 - From audio to visual context

Show the technical pipeline: five-second audio window -> 40 MFCC features -> Random Forest -> policy layer -> visual alert and event history.

### Slide 5 - What the prototype recognises

Display the eight classes as a balanced sound map. Visually distinguish the three alert-oriented classes from ordinary/context classes.

### Slide 6 - Baseline evidence

Lead with 81.25% validation accuracy and 80.93% macro F1. Add a compact bar chart for precision, recall, and F1 of door knock, glass breaking, and siren.

### Slide 7 - Why policy matters

Show that a classifier score alone is not an alert. Demonstrate configurable thresholds, priority, cooldown, snooze, acknowledgement, and event history. Mention wind/siren confusion as the reason false-alert controls are part of the product design.

### Slide 8 - Responsible prototype boundaries

State clearly: Fold 5 final evaluation is still pending; alert thresholds have not yet been selected through a fixed validation policy; real microphone and diverse-environment testing is still needed; the backend model hook is currently a placeholder. This transparency is part of the product credibility.

### Slide 9 - Roadmap and ask

Show the next steps:

1. Connect the trained model to the FastAPI inference endpoint.
2. Agree a threshold objective that balances missed important sounds against false alerts.
3. Freeze the model and policy, then run the reserved Fold 5 evaluation.
4. Test with real microphones, varied rooms, background noise, and accessibility users.
5. Build the visual client and measure alert usefulness, acknowledgement time, and user trust.

Close with the ask: support a focused pilot to turn a measured audio baseline into a reliable, user-configurable awareness tool.

## Demo script for the presenter

1. Upload or simulate a short clip.
2. Show the predicted sound and confidence.
3. Show the policy decision: alert, ignore, or suppress because of snooze/cooldown.
4. Show the visual prompt for a priority sound.
5. Acknowledge the event and open the event history.
6. Change a preference or snooze a sound to demonstrate user control.

For a live demo, label any simulated result clearly and do not imply that the current hard-coded backend prediction is the trained model output.

## Claims the deck may make

- HearHere is a working assistive prototype with a reproducible audio classification baseline.
- The current model reaches 81.25% accuracy and 80.93% macro F1 on the measured ESC-50 Fold 4 validation split.
- The product combines sound recognition with a user-facing alert policy, rather than exposing raw classifier scores alone.
- The design prioritises user control through thresholds, cooldowns, snooze, acknowledgement, and history.

## Claims to avoid

- Do not call HearHere a validated emergency-warning system.
- Do not describe Random Forest scores as calibrated real-world probabilities.
- Do not claim final test performance before Fold 5 is run.
- Do not claim live end-to-end model inference while the backend `predict()` placeholder remains.
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
