# Refrigerator AI Quality Inspection

A Streamlit dashboard that uses a YOLO11 Nano model to verify refrigerator interior components from uploaded or browser-camera images.

## Features

- Detects glass shelves, door balconies, fruit and vegetable boxes, egg trays, and ice-tray packs.
- Compares detected quantities with the required assembly checklist.
- Shows a pass or fail decision, component confidence, annotated images, and downloadable inspection history.
- Supports image uploads and browser-camera capture. The camera feature works on Streamlit Community Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Create an app at https://share.streamlit.io.
3. Select the repository and branch.
4. Set the main file path to `dashboard.py` and deploy.

The `best.pt` model file is included because the dashboard loads it at startup.

## Note

The app is a decision-support quality-control tool. Validate it on a sufficiently large, representative production dataset before using it as the sole source of factory pass/reject decisions.
