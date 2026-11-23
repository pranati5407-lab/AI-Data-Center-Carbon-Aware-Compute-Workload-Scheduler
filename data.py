"""Synthetic (but realistic-shaped) grid + workload data.

ASSUMPTIONS (state these in your report):
- Carbon intensity and prices are synthetic curves shaped like real grids
  (solar dip, wind noise, evening peak). Swap in real data via load_carbon_csv().
- Job power draw is sampled, not taken from a trace (cluster traces have no power data).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

SLOT_HOURS = 0.5          # each slot = 30 minutes
SLOTS_PER_DAY = 48
HORIZON = 96              # 48 hours
REGIONS = ["US-West", "EU-Central", "AP-South"]
HOME_REGION = 0           # where FIFO runs everything


@dataclass
class Job:
    id: int
    arrival: int      # earliest start slot
    duration: int     # slots
    power_kw: int
    deadline: int     # job must FINISH by this slot (exclusive end <= deadline)
    tier: str         # "urgent" | "batch"


def make_grid(seed: int = 7):
    """Returns carbon[r, t] in gCO2/kWh and price[r, t] in $/kWh."""
    rng = np.random.default_rng(seed)
    t = np.arange(HORIZON)
    hour = (t * SLOT_HOURS) % 24
    carbon = np.zeros((len(REGIONS), HORIZON))
    price = np.zeros((len(REGIONS), HORIZON))

    # local-time offsets so each region's clean hours fall at different times
    offsets = [0, 8, 13]
    for r, off in enumerate(offsets):
        h = (hour + off) % 24
        solar = np.exp(-((h - 13) ** 2) / (2 * 3.0 ** 2))          # midday dip
        evening = np.exp(-((h - 19) ** 2) / (2 * 2.0 ** 2))        # evening peak
        wind = np.convolve(rng.normal(0, 1, HORIZON + 10), np.ones(6) / 6, "same")[:HORIZON]
        if r == 0:    # solar-heavy
            base = 380 - 220 * solar + 90 * evening + 25 * wind
        elif r == 1:  # wind + mixed
            base = 300 - 60 * solar + 50 * evening + 70 * wind
        else:         # coal-heavy, flat
            base = 620 - 40 * solar + 60 * evening + 20 * wind
        carbon[r] = np.clip(base, 40, None)
        # time-of-use tariff: cheap night, expensive evening peak
        tariff = 0.08 + 0.07 * evening + 0.03 * (h > 8) * (h < 22)
        price[r] = tariff * (1 + 0.1 * r) + rng.normal(0, 0.003, HORIZON)
    return carbon, np.clip(price, 0.03, None)


def load_carbon_csv(path: str):
    """Real data: first len(REGIONS) columns (any names), rows = half-hour slots -> array [R, HORIZON].
    Column 0 becomes the home region. Plots keep the generic region labels."""
    df = pd.read_csv(path)
    arr = df.iloc[:HORIZON, :len(REGIONS)].to_numpy(dtype=float).T
    if arr.shape[1] < HORIZON:
        raise ValueError(f"need {HORIZON} rows, got {arr.shape[1]}")
    return arr


def make_jobs(n: int = 80, seed: int = 11, urgent_frac: float = 0.25, batch_dur=None):
    """batch_dur=(lo, hi) overrides the duration range of batch jobs (e.g. long training runs)."""
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n):
        arrival = int(rng.integers(0, SLOTS_PER_DAY))
        duration = int(rng.integers(1, 9))
        power = int(rng.integers(5, 51))
        is_urgent = rng.random() < urgent_frac
        if is_urgent:
            tier, slack = "urgent", int(rng.integers(0, 3))
        else:
            tier, slack = "batch", int(rng.integers(12, 44))
            if batch_dur:
                duration = int(rng.integers(*batch_dur))
        deadline = min(HORIZON, arrival + duration + slack)
        jobs.append(Job(i, arrival, duration, power, deadline, tier))
    return jobs
