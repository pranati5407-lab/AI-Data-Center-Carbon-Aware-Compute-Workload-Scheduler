"""Robustness check: repeat the static comparison over 10 random grids + job sets."""
import numpy as np
import pandas as pd
from data import make_grid, make_jobs
from schedulers import fifo, greedy, greedy_home, milp, evaluate

ALPHA, SEEDS = 0.7, range(1, 11)
rows = []
for sd in SEEDS:
    c, p = make_grid(seed=sd)
    jobs = make_jobs(n=80, seed=sd)
    base = evaluate(jobs, fifo(jobs, c, p), c, p)
    g = evaluate(jobs, greedy(jobs, c, p, ALPHA), c, p)
    h = evaluate(jobs, greedy_home(jobs, c, p, ALPHA), c, p)
    s, info = milp(jobs, c, p, ALPHA, time_limit=8)
    m = evaluate(jobs, s, c, p)
    rows.append({"seed": sd,
                 "timeshift_only_co2_saved_%": 100 * (1 - h["co2_kg"] / base["co2_kg"]),
                 "timeshift_only_cost_saved_%": 100 * (1 - h["cost_usd"] / base["cost_usd"]),
                 "timeshift_only_sla_misses": h["sla_violations"],
                 "opt_gap_%": 100 * info["gap"], "opt_status": info["status"],
                 "greedy_co2_saved_%": 100 * (1 - g["co2_kg"] / base["co2_kg"]),
                 "opt_co2_saved_%": 100 * (1 - m["co2_kg"] / base["co2_kg"]),
                 "greedy_cost_saved_%": 100 * (1 - g["cost_usd"] / base["cost_usd"]),
                 "opt_cost_saved_%": 100 * (1 - m["cost_usd"] / base["cost_usd"]),
                 "fifo_sla_misses": base["sla_violations"],
                 "greedy_sla_misses": g["sla_violations"], "opt_sla_misses": m["sla_violations"]})
    print(f"seed {sd}: greedy {rows[-1]['greedy_co2_saved_%']:.1f}%  optimizer {rows[-1]['opt_co2_saved_%']:.1f}%", flush=True)
df = pd.DataFrame(rows)
df.to_csv("multiseed_results.csv", index=False)
print("\nMean ± std over", len(df), "seeds")
for col in [c for c in df.columns[1:] if c != "opt_status"]:
    print(f"  {col:22s} {df[col].mean():6.1f} ± {df[col].std():.1f}")
print("  solver status:", df["opt_status"].value_counts().to_dict())
print("  optimizer beats greedy on CO2 in", int((df['opt_co2_saved_%'] > df['greedy_co2_saved_%']).sum()), "of", len(df), "seeds")
