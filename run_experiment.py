import sys
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_grid, make_jobs, load_carbon_csv, REGIONS, SLOT_HOURS
from schedulers import fifo, greedy, milp, evaluate

carbon, price = make_grid()
if len(sys.argv) > 1:                       # python run_experiment.py carbon_real.csv
    carbon = load_carbon_csv(sys.argv[1])   # real carbon, synthetic prices
    print('Using REAL carbon data from', sys.argv[1])
jobs = make_jobs(n=80)
ALPHA = 0.7   # weight on carbon vs cost

rows, schedules = [], {}
for name, fn in [("FIFO", fifo), ("Greedy", greedy), ("MILP", milp)]:
    t0 = time.time()
    out = fn(jobs, carbon, price, ALPHA)
    sched, info = out if isinstance(out, tuple) else (out, {})
    schedules[name] = sched
    m = evaluate(jobs, sched, carbon, price)
    m.update(scheduler=name, solve_s=round(time.time() - t0, 2), **info)
    rows.append(m)

df = pd.DataFrame(rows).set_index("scheduler")
base = df.loc["FIFO"]
df["co2_saved_%"] = (100 * (1 - df.co2_kg / base.co2_kg)).round(1)
df["cost_saved_%"] = (100 * (1 - df.cost_usd / base.cost_usd)).round(1)
print(f"\nalpha = {ALPHA}, jobs = {len(jobs)}\n")
print(df.round(2).to_string())

# ---- alpha sweep (carbon vs cost Pareto) ----
pts = []
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    s, _ = milp(jobs, carbon, price, a, time_limit=20)
    m = evaluate(jobs, s, carbon, price)
    pts.append((a, m["co2_kg"], m["cost_usd"]))
sweep = pd.DataFrame(pts, columns=["alpha", "co2_kg", "cost_usd"])
print("\nAlpha sweep (MILP):\n", sweep.round(2).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(sweep.cost_usd, sweep.co2_kg, "o-", label="MILP (alpha sweep)")
for a, c, p in pts:
    ax[0].annotate(f"α={a}", (p, c), textcoords="offset points", xytext=(5, 5), fontsize=8)
for n in ["FIFO", "Greedy"]:
    ax[0].scatter(df.loc[n, "cost_usd"], df.loc[n, "co2_kg"], s=70, label=n)
ax[0].set_xlabel("Electricity cost ($)"); ax[0].set_ylabel("CO₂ (kg)")
ax[0].set_title("Carbon vs cost trade-off"); ax[0].legend()

for r, name in enumerate(REGIONS):
    ax[1].plot(carbon[r], label=name)
ax[1].set_xlabel("slot (30 min)"); ax[1].set_ylabel("gCO₂/kWh")
ax[1].set_title("Synthetic grid carbon intensity"); ax[1].legend()
plt.tight_layout(); plt.savefig("results.png", dpi=140)
print("\nSaved results.png")
