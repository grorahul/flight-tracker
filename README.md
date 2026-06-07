# ✈️ Flight On-Time Departure Tracker

Find the last 10 departure history of any flight and predict on-time probability using OpenSky Network and Machine Learning.

## Features
- Search any flight callsign e.g. SQ285, EK432, QF1
- View last 10 departures with date, time, airport and delay
- ML-based on-time probability prediction using Random Forest
- On-time rate summary across recent history
- 100% free powered by OpenSky Network

## How It Works
1. User enters a flight callsign
2. Backend fetches historical departures from OpenSky Network
3. ML model predicts on-time probability based on patterns
4. Frontend displays last 10 results with on-time status

## Tech Stack
- Backend: FastAPI hosted on Render.com
- Frontend: HTML/CSS/JS hosted on GitHub Pages
- ML: scikit-learn Random Forest
- Data: OpenSky Network free API

## Live Demo
- Frontend: https://YOUR_USERNAME.github.io/flight-tracker
- API: https://YOUR_APP.onrender.com

## Local Setup
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Set environment variables: OPENSKY_USER and OPENSKY_PASS
4. Fetch data: python backend/data_fetcher.py --callsign SQ285 --days 60
5. Train model: python backend/model_trainer.py --callsign SQ285
6. Run API: uvicorn backend.main:app --reload --port 8000

## Data Note
OpenSky does not provide official scheduled times so delay is a proxy based on historical median departure time.
