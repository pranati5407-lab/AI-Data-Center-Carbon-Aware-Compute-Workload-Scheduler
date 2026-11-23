"""Download the next 48h of REAL half-hourly carbon intensity from the free UK Carbon Intensity API
(no API key; CC BY 4.0) for three GB regions and save carbon_real.csv.

    python fetch_real_carbon.py            # then:  python run_experiment.py carbon_real.csv

Column order matters: column 0 is the simulator's HOME region (where FIFO runs), so we put a
fairly carbon-intensive region first and a cleaner (wind-heavy) region second.
All three regions share one timezone, so spatial differences come only from generation mix,
not from time-of-day offsets as in the synthetic data. Prices stay synthetic.
"""
import csv
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = "https://api.carbonintensity.org.uk"
REGIONS = [(12, "South England"), (1, "North Scotland"), (9, "East Midlands")]   # (region id, name)
SLOTS = 96


def parse_region(payload, slots=SLOTS):
    """Pull the forecast gCO2/kWh series out of the API's JSON (data -> data[] -> intensity)."""
    rows = sorted(payload["data"]["data"], key=lambda r: r["from"])
    vals = []
    for r in rows:
        i = r["intensity"]
        v = i.get("actual") if i.get("actual") is not None else i.get("forecast")
        if v is not None:
            vals.append(float(v))
    if len(vals) < slots:
        raise ValueError(f"only {len(vals)} half-hours returned, need {slots}")
    return vals[:slots]


def fetch(region_id, start):
    url = f"{BASE}/regional/intensity/{start}/fw48h/regionid/{region_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main(out="carbon_real.csv"):
    now = datetime.now(timezone.utc)
    now -= timedelta(minutes=now.minute % 30, seconds=now.second, microseconds=now.microsecond)
    start = now.strftime("%Y-%m-%dT%H:%MZ")
    cols = {name: parse_region(fetch(rid, start)) for rid, name in REGIONS}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols.keys())
        w.writerows(zip(*cols.values()))
    print(f"Saved {out} ({SLOTS} half-hour slots from {start}) for {list(cols)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
