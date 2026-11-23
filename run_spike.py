"""Phase 2: unexpected carbon spike in US-West. Does adaptive re-planning + preemption help?"""
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_grid, make_jobs, REGIONS
from schedulers import fifo, milp, evaluate
from online import run_online, evaluate_online, load_profile

ALPHA = 0.7
forecast, price = make_grid()
jobs = make_jobs(n=80)

SPIKE_REGION, SPIKE_START, SPIKE_END, SPIKE_ADD = 0, 24, 36, 450     # +450 gCO2/kWh for 6 hours
actual = forecast.copy()
actual[SPIKE_REGION, SPIKE_START:SPIKE_END] += SPIKE_ADD
spike = (SPIKE_REGION, SPIKE_START, SPIKE_END)

rows, segs = [], {}

def record(name, metrics, secs, sg):
    metrics.update(scheduler=name, solve_s=round(secs, 1))
    rows.append(metrics)
    segs[name] = sg

t0 = time.time(); s = fifo(jobs, forecast, price)
record("FIFO", {**evaluate(jobs, s, actual, price), "preemptions": 0}, time.time() - t0,
       {j: [(r, st, next(x.duration for x in jobs if x.id == j))] for j, (r, st) in s.items()})

t0 = time.time(); s, _ = milp(jobs, forecast, price, ALPHA, time_limit=20)
record("Offline plan (blind to spike)", {**evaluate(jobs, s, actual, price), "preemptions": 0}, time.time() - t0,
       {j: [(r, st, next(x.duration for x in jobs if x.id == j))] for j, (r, st) in s.items()})

for name, aware, pre in [("Online re-solve (blind to spike)", False, False),
                         ("Online aware, no preemption", True, False),
                         ("Online aware + preemption", True, True)]:
    t0 = time.time()
    st = run_online(jobs, forecast, actual, price, ALPHA, spike=spike, aware=aware, preempt=pre, verbose=pre)
    record(name, evaluate_online(st, actual, price), time.time() - t0,
           {j: [tuple(x) for x in v.segs] for j, v in st.items()})

df = pd.DataFrame(rows).set_index("scheduler")
base = df.loc["FIFO"]
df["co2_vs_FIFO_%"] = (100 * (1 - df.co2_kg / base.co2_kg)).round(1)
df["cost_vs_FIFO_%"] = (100 * (1 - df.cost_usd / base.cost_usd)).round(1)
print(f"\nSpike: {REGIONS[SPIKE_REGION]} +{SPIKE_ADD} gCO2/kWh, slots {SPIKE_START}-{SPIKE_END} (unforecast)\n")
print(df[["co2_kg", "cost_usd", "sla_violations", "unscheduled", "preemptions",
          "co2_vs_FIFO_%", "cost_vs_FIFO_%", "solve_s"]].round(1).to_string())

# ---------------- plots ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
x = range(len(forecast[0]))
ax[0].plot(x, forecast[SPIKE_REGION], label="forecast"); ax[0].plot(x, actual[SPIKE_REGION], label="actual")
ax[0].axvspan(SPIKE_START, SPIKE_END, alpha=0.15, color="red")
ax[0].set_title(f"{REGIONS[SPIKE_REGION]} carbon: unexpected spike"); ax[0].set_xlabel("slot (30 min)")
ax[0].set_ylabel("gCO₂/kWh"); ax[0].legend()

for name in ["Offline plan (blind to spike)", "Online aware + preemption"]:
    ax[1].step(x, load_profile(segs[name], jobs, SPIKE_REGION), where="post", label=name)
ax[1].axvspan(SPIKE_START, SPIKE_END, alpha=0.15, color="red")
ax[1].set_title(f"Load in {REGIONS[SPIKE_REGION]} (kW)"); ax[1].set_xlabel("slot"); ax[1].legend(fontsize=7)

ax[2].barh(df.index, df.co2_kg, color=["#888", "#c55", "#e90", "#6a9", "#2a7"])
ax[2].set_xlabel("Total CO₂ under actual grid (kg)"); ax[2].set_title("Result"); ax[2].invert_yaxis()
plt.tight_layout(); plt.savefig("results_spike.png", dpi=140)
print("\nSaved results_spike.png")
