#!/usr/bin/env python3
"""
Test script — verifies API access for the three candidate T5 augmentation
sources: PredictHQ (events), 511NY (traffic/incidents), MTA Service Alerts.

Reads API keys from environment variables so nothing sensitive is hardcoded:
  PREDICTHQ_TOKEN  — from control.predicthq.com -> API Tools -> API Tokens
  NY511_API_KEY    — from 511ny.org, after Developer Access approval

MTA Service Alerts needs no key for light testing (public Socrata endpoint).

Usage:
  export PREDICTHQ_TOKEN="your_token_here"
  export NY511_API_KEY="your_key_here"     # once approved
  python3 test_apis.py
"""

import os
import requests

NYC_PLACE_ID = "5128581"  # PredictHQ's place.scope ID for New York City


def test_predicthq():
    token = os.environ.get("PREDICTHQ_TOKEN")
    if not token:
        return "SKIPPED", "PREDICTHQ_TOKEN not set"

    try:
        response = requests.get(
            url="https://api.predicthq.com/v1/events/",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={
                "place.scope": NYC_PLACE_ID,
                "category": "concerts,sports,conferences",
                "limit": 3,
            },
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            count = data.get("count", 0)
            sample = [e.get("title") for e in data.get("results", [])[:3]]
            return "OK", f"{count} total matches, sample: {sample}"
        else:
            return "FAILED", f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return "ERROR", str(e)


def test_511ny():
    key = os.environ.get("NY511_API_KEY")
    if not key:
        return "SKIPPED", "NY511_API_KEY not set (may still be awaiting approval)"

    try:
        response = requests.get(
            url="https://511ny.org/api/getevents",
            params={"key": key, "format": "json"},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            n = len(data) if isinstance(data, list) else "?"
            return "OK", f"{n} events returned"
        else:
            return "FAILED", f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return "ERROR", str(e)


def test_mta_alerts():
    try:
        response = requests.get(
            url="https://data.ny.gov/resource/7kct-peq7.json",
            params={"$limit": 3},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            sample_cols = list(data[0].keys()) if data else []
            return "OK", f"{len(data)} rows returned, columns: {sample_cols}"
        else:
            return "FAILED", f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return "ERROR", str(e)


def main():
    tests = [
        ("PredictHQ (events)", test_predicthq),
        ("511NY (traffic/incidents)", test_511ny),
        ("MTA Service Alerts", test_mta_alerts),
    ]

    print("=" * 70)
    print("API ACCESS TEST")
    print("=" * 70)

    results = []
    for name, fn in tests:
        status, detail = fn()
        results.append((name, status, detail))
        print(f"\n[{status}] {name}")
        print(f"  {detail}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, status, _ in results:
        print(f"  {status:8s} — {name}")


if __name__ == "__main__":
    main()