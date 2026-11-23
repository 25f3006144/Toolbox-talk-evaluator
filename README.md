# ToolboxAI – AI-Powered Toolbox Talk Evaluator

**An AI-powered workplace safety evaluation system that analyzes toolbox talk sessions using Computer Vision, Speech-to-Text, and automated evaluation techniques.**

## About the Project

ToolboxAI is a Python-based application designed to evaluate workplace safety training sessions (Toolbox Talks) using video and audio analysis.

The system extracts important safety-related parameters, evaluates participant engagement and PPE compliance, and generates a structured evaluation report.

## Key Features

- **Video Processing:** Video frame extraction, resizing, and shot boundary detection.
- **Audio Processing:** Audio extraction and preprocessing.
- **Speech-to-Text:** Automatic speech transcription using Faster-Whisper.
- **Computer Vision:** Person detection, face detection, attention analysis, PPE detection, and scene analysis.
- **Parameter Extraction:** Extracts important safety training parameters from video and audio.
- **Scoring Engine:** Evaluates the extracted parameters and calculates evaluation scores.
- **Evaluation Report:** Generates a structured report with scores and improvement suggestions.
- **Web Dashboard:** Provides a user interface to interact with the application.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Video processing and computer vision |
| YOLO11 | Object and person detection |
| YuNet | Face detection |
| Faster-Whisper | Speech-to-text transcription |
| Flask | Web application |
| HTML/CSS | Frontend interface |
| Git & GitHub | Version control |

## Project Structure

```text
Toolbox-talk-evaluator/
│
├── main.py
├── app.py
├── generate_report.py
├── inspect_cv_output.py
│
├── src/
│   ├── audio/
│   ├── speech/
│   ├── video/
│   ├── cv/
│   └── evaluation/
│
├── templates/
│   └── index.html
│
├── tests/
├── checks/
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/25f3006144/Toolbox-talk-evaluator.git
cd Toolbox-talk-evaluator
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Install the Python packages required by the project.

## Running the Project

Run the main evaluation pipeline:

```bash
python main.py
```

To launch the web application:

```bash
python app.py
```

## Evaluation Workflow

1. Upload or provide a toolbox talk video.
2. Process video and audio.
3. Extract speech using Speech-to-Text.
4. Analyze participants, attention, PPE, and scenes.
5. Extract safety-related parameters.
6. Calculate evaluation scores.
7. Generate the final evaluation report.

## Project Goals

- Automate workplace safety training evaluation.
- Improve the consistency of toolbox talk assessments.
- Identify areas that need improvement.
- Explore practical applications of Artificial Intelligence in workplace safety.

## Project Status

**Development:** Ongoing

This project is being developed as a practical application of Python, Artificial Intelligence, Computer Vision, and Natural Language Processing.

---

**Developed as an AI-based workplace safety evaluation project.**
