"""
data_fetcher.py
---------------
Fetches historical departure data for a given callsign from the OpenSky Network REST API.
Requires a free account at https://opensky-network.org

Usage:
    python data_fetcher.py --callsign SQ285 --days 30
"""

import requests
import time
import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

OPENSKY_BASE = "https://opensky-network.org/api"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

import os
OPENSKY_USER = os.getenv("OPENSKY_USER", "")
OPENSKY_PASS = os.getenv("OPENSKY_PASS", "")


def get_auth():
    if OPENSKY_USER and OPENSKY_PASS:
        return (OPENSKY_USER, OPENSKY_PASS)
    print("No credentials set — using anonymous access.")
    return None


def fetch_flights_by_callsign(callsign: str, days_back: int = 30) -> list[dict]:
    auth = get_auth()
    callsign_padded = callsign.upper().ljust(8)

    now = int(datetime.now(timezone.utc).timestamp())
    start = now - (days_back * 86400)

    all_flights = []
    chunk_seconds = 7 * 86400
    cursor = start

    print(f"Fetching flights for callsign: {callsign_padded.strip()} ...")

    while cursor < now:
        end_chunk = min(cursor + chunk_seconds, now)
        url = f"{OPENSKY_BASE}/flights/all"
        params = {"begin": cursor, "end": end_chunk}

        try:
            resp = requests.get(url, params=params, auth=auth, timeout=20)

            if resp.status_code == 200:
                flights = resp.json() or []
                matched = [
                    f for f in flights
                    if f.get("callsign", "").strip().upper() == callsign.upper()
                ]
                all_flights.extend(matched)
                print(f"   {datetime.fromtimestamp(cursor).date()} → "
                      f"{datetime.fromtimestamp(end_chunk).date()} : "
                      f"{len(matched)} flights found")

            elif resp.status_code == 404:
                print(f"   No data for window ending {datetime.fromtimestamp(end_chunk).date()}")

            elif resp.status_code == 429:
                print("   Rate limited — sleeping 60s ...")
                time.sleep(60)
                continue

            else:
                print(f"   HTTP {resp.status_code}: {resp.text[:100]}")

        except requests.RequestException as e:
            print(f"   Request error: {e}")

        cursor = end_chunk
        time.sleep(1)

    return all_flights


def parse_flight(raw: dict) -> dict:
    first_seen = raw.get("firstSeen")
    last_seen = raw.get("lastSeen")

    dt_dep = datetime.fromtimestamp(first_seen, tz=timezone.utc) if first_seen else None
    dt_arr = datetime.fromtimestamp(last_seen, tz=timezone.utc) if last_seen else None

    return {
        "callsign": raw.get("callsign", "").strip(),
        "icao24": raw.get("icao24", ""),
        "departure_airport": raw.get("estDepartureAirport", ""),
        "arrival_airport": raw.get("estArrivalAirport", ""),
        "first_seen_unix": first_seen,
        "last_seen_unix": last_seen,
        "departure_date": dt_dep.strftime("%Y-%m-%d") if dt_dep else None,
        "departure_time_utc": dt_dep.strftime("%H:%M") if dt_dep else None,
        "departure_hour": dt_dep.hour if dt_dep else None,
        "departure_minute": dt_dep.minute if dt_dep else None,
        "day_of_week": dt_dep.weekday() if dt_dep else None,
        "day_of_week_name": dt_dep.strftime("%A") if dt_dep else None,
        "month": dt_dep.month if dt_dep else None,
        "flight_duration_minutes": (
            round((last_seen - first_seen) / 60, 1)
            if first_seen and last_seen else None
        ),
    }


def compute_on_time(flights: list[dict]) -> list[dict]:
    import statistics

    minutes = [f["departure_minute"] for f in flights if f["departure_minute"] is not None]
    hours = [f["departure_hour"] for f in flights if f["departure_hour"] is not None]

    if not minutes:
        return flights

    expected_hour = round(statistics.median(hours))
    expected_minute = round(statistics.median(minutes))
    print(f"Inferred scheduled departure: ~{expected_hour:02d}:{expected_minute:02d} UTC")

    for f in flights:
        if f["departure_hour"] is not None:
            actual_total = f["departure_hour"] * 60 + f["departure_minute"]
            expected_total = expected_hour * 60 + expected_minute
            delay_minutes = actual_total - expected_total

            if delay_minutes > 720:
                delay_minutes -= 1440
            elif delay_minutes < -720:
                delay_minutes += 1440

            f["delay_minutes"] = delay_minutes
            f["on_time"] = 1 if abs(delay_minutes) <= 15 else 0
        else:
            f["delay_minutes"] = None
            f["on_time"] = None

    return flights


def save_data(flights: list[dict], callsign: str):
    out_path = DATA_DIR / f"{callsign.upper()}_history.json"
    with open(out_path, "w") as f:
        json.dump(flights, f, indent=2)
    print(f"Saved {len(flights)} flights to {out_path}")
    return out_path


def fetch_and_save(callsign: str, days_back: int = 30) -> list[dict]:
    raw_flights = fetch_flights_by_callsign(callsign, days_back)

    if not raw_flights:
        print(f"No flights found for {callsign}.")
        return []

    parsed = [parse_flight(f) for f in raw_flights]
    labelled = compute_on_time(parsed)
    save_data(labelled, callsign)

    on_time_count = sum(1 for f in labelled if f.get("on_time") == 1)
    print(f"\nSummary for {callsign.upper()}:")
    print(f"   Total flights found : {len(labelled)}")
    print(f"   On time (±15 min)   : {on_time_count}")
    print(f"   Delayed             : {len(labelled) - on_time_count}")

    return labelled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--callsign", required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    fetch_and_save(args.callsign, args.days)  
