# HearHere Solution Architecture

This diagram shows the implemented local flow from an audio input to a visual alert. The browser and backend run on the same machine during local development; the HTTP contract remains the same when the backend is hosted elsewhere.

```mermaid
flowchart LR
    subgraph Client["Browser client - Vite + React"]
        Input["Microphone or WAV upload"]
        Window["Mono 16 kHz<br/>5-second WAV windows"]
        UI["Results, visual alert,<br/>settings and event log"]
        Input --> Window
    end

    subgraph Service["FastAPI backend - 127.0.0.1:8000"]
        Endpoint["POST /classify"]
        Contract["Validate upload<br/>size, WAV format and duration"]
        Classifier["HearHereClassifier"]
        Policy["Alert policy<br/>thresholds, preferences,<br/>2-of-3 monitoring confirmation,<br/>cooldown and snooze"]
        Response["JSON response<br/>label, scores, action and event id"]
        Endpoint --> Contract --> Classifier --> Policy --> Response
    end

    subgraph Model["Validated model runtime"]
        Prepare["Shared preprocessing<br/>mono, resample, zero-pad"]
        Backbone["Frozen PANNs CNN14<br/>2048-value embedding"]
        Head["Standardized Logistic Regression<br/>ten class scores"]
        Prepare --> Backbone --> Head
    end

    subgraph State["Local artifacts and state"]
        Artifacts["models/<br/>model + metadata + thresholds"]
        Events[("events.db<br/>event history")]
        Checkpoint[("User cache<br/>verified PANNs checkpoint")]
    end

    Window -->|multipart WAV| Endpoint
    Classifier --> Prepare
    Artifacts --> Classifier
    Checkpoint --> Backbone
    Policy --> Events
    Response --> UI
    Events --> UI
    UI -.->|events, acknowledgement,<br/>preferences and snooze| Service

    subgraph Training["Offline training and evaluation"]
        Dataset["ESC-50 folds 1-3"] --> Train["scripts/train.py"]
        Validation["Fold 4 validation<br/>and threshold selection"] --> Train
        Train --> Reports["reports/<br/>metrics and confusion matrix"]
        Final["Fold 5 final evaluation<br/>after model lock"] -.-> Reports
    end

    Train -.-> Artifacts
```

The request path is `browser -> POST /classify -> shared preprocessing -> PANNs embedding -> Logistic Regression scores -> backend alert policy -> browser response`. The backend owns alert decisions and event state; the frontend displays the returned `action` and scores.

Model scores are uncalibrated classifier scores rather than real-world event probabilities. Fold 4 selects thresholds, while Fold 5 is reserved for the final held-out report. The current model recognises ten classes and keeps the six alert-oriented classes configurable through the backend policy.

Related contracts:

- [Current API contract](classify_api.md)
- [Model artifact contract](model-artifacts.md)
- [Project structure](project-structure.md)
