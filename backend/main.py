"""
main.py
-------
FastAPI backend for the Flight On-Time Departure Tracker.

Run locally:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import statistics

from backend.data_fetcher import fetch_flights_by_callsign, parse_flight, compute_on_time
from backend.model_trainer import predict_single

app = FastAPI(
    title="Flight On-Time Tracker",
    description="Fetch last 10 departures + on-time predictions using OpenSky Network",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"


class FlightRecord(BaseModel):
    callsign: str
    departure_date: str | None
    departure_time_utc: str | None
    departure_airport: str | None
    arrival_airport: str | None
    delay_minutes: float | None
    on_time: int | None
    on_time_label: str
    day_of_week_name: str | None
    flight_duration_minutes: float | None
    prediction: dict | None


class FlightHistoryResponse(BaseModel):
    callsign: str
    fetched_at: str
    total_records: int
    on_time_rate: str
    inferred_scheduled_utc: str | None
    flights: list[FlightRecord]


@app.get("/")
def root():
    return {"status": "ok", "message": "Flight Tracker API is running"}


@app.get("/flight/{callsign}", response_model=FlightHistoryResponse)
def get_flight_history(callsign: str, days: int = 60):
    callsign = callsign.upper().strip()
    days = min(days, 90)

    cache_path = DATA_DIR / f"{callsign}_history.json"
    flights_parsed = []

    if cache_path.exists():
        mtime = cache_path.stat().st_mtime
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours < 12:
            with open(cache_path) as f:
                flights_parsed = json.load(f)

    if not flights_parsed:
        raw = fetch_flights_by_callsign(callsign, days_back=days)
        if not raw:
            raise HTTPException(
                status_code=404,
                detail=f"No flights found for '{callsign}'. Try a different callsign or more days."
            )
        flights_parsed = [parse_flight(f) for f in raw]
        flights_parsed = compute_on_time(flights_parsed)

        DATA_DIR.mkdir(exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(flights_parsed, f, indent=2)

    valid = [f for f in flights_parsed if f.get("departure_date")]
    valid.sort(key=lambda x: x["departure_date"], reverse=True)
    last_10 = valid[:10]

    if not last_10:
        raise HTTPException(status_code=404, detail="No valid departure records found.")

    hours = [f["departure_hour"] for f in valid if f.get("departure_hour") is not None]
    mins = [f["departure_minute"] for f in valid if f.get("departure_minute") is not None]
    inferred_sched = None
    if hours and mins:
        inferred_sched = f"{round(statistics.median(hours)):02d}:{round(statistics.median(mins)):02d} UTC"

    model_exists = (MODEL_DIR / f"{callsign}_model.pkl").exists()
    records = []

    for f in last_10:
        prediction = None
        if model_exists and f.get("departure_hour") is not None:
            try:
                prediction = predict_single(
                    callsign=callsign,
                    day_of_week=f["day_of_week"],
                    hour=f["departure_hour"],
                    minute=f["departure_minute"],
                    month=f["month"],
                    airport=f.get("departure_airport") or "UNKNOWN"
                )
            except Exception as e:
                prediction = {"error": str(e)}

        records.append(FlightRecord(
            callsign=f["callsign"],
            departure_date=f.get("departure_date"),
            departure_time_utc=f.get("departure_time_utc"),
            departure_airport=f.get("departure_airport"),
            arrival_airport=f.get("arrival_airport"),
            delay_minutes=f.get("delay_minutes"),
            on_time=f.get("on_time"),
            on_time_label="On Time" if f.get("on_time") == 1 else "Delayed",
            day_of_week_name=f.get("day_of_week_name"),
            flight_duration_minutes=f.get("flight_duration_minutes"),
            prediction=prediction,
        ))

    on_time_count = sum(1 for r in records if r.on_time == 1)
    rate = f"{on_time_count}/{len(records)} on time"

    return FlightHistoryResponse(
        callsign=callsign,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_records=len(valid),
        on_time_rate=rate,
        inferred_scheduled_utc=inferred_sched,
        flights=records,
    )


@app.get("/flight/{callsign}/train")
def trigger_training(callsign: str):
    from backend.model_trainer import train
    callsign = callsign.upper().strip()
    try:
        result = train(callsign)
        if result:
            return {"status": "success", "message": f"Model trained for {callsign}"}
        return {"status": "warning", "message": "Training skipped — check data quality"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
