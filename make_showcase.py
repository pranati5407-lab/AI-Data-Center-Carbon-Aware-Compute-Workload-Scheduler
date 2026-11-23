"""Presentation-ready figures -> showcase/*.png"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import make_grid, make_jobs, REGIONS, HORIZON
from schedulers import fifo, milp
from online import load_profile

os.makedirs("showcase", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREY, GREEN, ORANGE, BLUE = "#8a8a8a", "#1f9d6b", "#e8890c", "#3b6fd4"

# ---------- Fig 1: headline CO2 savings across scenarios ----------
# values from run_multiseed.py, run_spike.py, run_long_jobs.py (final code)
labels = ["Static scheduling\n(10-seed mean)", "Grid spike:\nplan blind to spike", "Grid spike:\nadaptive re-planning",
          "Long jobs:\nadaptive, no preemption", "Long jobs:\nadaptive + preemption"]
vals = [27.2, 29.5, 39.0, 33.3, 40.9]
cols = [BLUE, GREY, GREEN, GREY, GREEN]
fig, ax = plt.subplots(figsize=(9, 4.6))
b = ax.barh(labels, vals, color=cols); ax.invert_yaxis()
for r, v in zip(b, vals):
    ax.text(v + 0.6, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va="center", fontweight="bold")
ax.set_xlim(0, 48); ax.set_xlabel("CO₂ reduction vs FIFO baseline (%)")
ax.set_title("Carbon-aware scheduling cuts CO₂ by 27–41% with zero SLA misses", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("showcase/fig1_headline_savings.png", dpi=170); plt.close()

# ---------- Fig 2: where the load goes (FIFO vs optimizer) ----------
c, p = make_grid(); jobs = make_jobs(80); dur = {j.id: j.duration for j in jobs}
sf = fifo(jobs, c, p)
so, _ = milp(jobs, c, p, 0.7, time_limit=15)
to_segs = lambda s: {j: [(r, st, dur[j])] for j, (r, st) in s.items()}
fig, axs = plt.subplots(3, 2, figsize=(13, 7.5), sharex=True)
x = np.arange(HORIZON)
for col, (name, s) in enumerate([("FIFO: run on arrival, one region", sf), ("Carbon-aware optimizer", so)]):
    segs = to_segs(s)
    for r in range(3):
        ax = axs[r, col]
        ax.fill_between(x, load_profile(segs, jobs, r), step="post", color=ORANGE if col == 0 else GREEN, alpha=0.85)
        ax.set_ylim(0, 230); ax.set_ylabel(f"{REGIONS[r]}\nload (kW)")
        ax2 = ax.twinx(); ax2.plot(x, c[r], color="k", lw=1.2); ax2.set_ylim(0, 700)
        ax2.spines["top"].set_visible(False)
        if col == 1: ax2.set_ylabel("gCO₂/kWh")
        else: ax2.set_yticks([])
    axs[0, col].set_title(name, fontweight="bold")
    axs[2, col].set_xlabel("time (30-min slots over 48 h)")
fig.suptitle("Same jobs, same deadlines: the optimizer moves load into low-carbon valleys (black line = grid carbon)", y=0.995, fontweight="bold")
plt.tight_layout(); plt.savefig("showcase/fig2_load_vs_carbon.png", dpi=170); plt.close()

# ---------- Fig 3: robustness over 10 seeds ----------
df = pd.read_csv("multiseed_results.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax[0].boxplot([df["greedy_co2_saved_%"], df["opt_co2_saved_%"]], tick_labels=["Greedy", "Optimizer"], widths=0.5,
              boxprops=dict(color=BLUE), medianprops=dict(color=GREEN, lw=2))
for i, col in enumerate(["greedy_co2_saved_%", "opt_co2_saved_%"], start=1):
    ax[0].scatter(np.random.default_rng(0).normal(i, 0.05, len(df)), df[col], color=BLUE, alpha=0.6, s=18)
ax[0].set_ylabel("CO₂ reduction vs FIFO (%)"); ax[0].set_title("10 random grids + job sets", fontweight="bold")
ax[1].bar(["FIFO", "Greedy", "Optimizer"], [df.fifo_sla_misses.mean(), df.greedy_sla_misses.mean(), df.opt_sla_misses.mean()],
          color=[ORANGE, GREEN, GREEN])
ax[1].set_ylabel("avg missed deadlines (of 80 jobs)"); ax[1].set_title("SLA compliance", fontweight="bold")
plt.tight_layout(); plt.savefig("showcase/fig3_multiseed_robustness.png", dpi=170); plt.close()

# ---------- Fig 4: carbon vs cost frontier ----------
a = [0, .25, .5, .75, 1]
co2 = [1680.65, 1355.72, 1062.43, 950.51, 938.25]; cost = [371.66, 380.50, 430.86, 478.42, 497.19]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(cost, co2, "o-", color=GREEN, lw=2, label="Optimizer (α sweep)")
for ai, ci, pi in zip(a, co2, cost):
    ax.annotate(f"α={ai}", (pi, ci), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.scatter([492.57], [1415.58], s=90, color=ORANGE, label="FIFO", zorder=3)
ax.scatter([470.37], [980.50], s=90, color=BLUE, marker="s", label="Greedy (α=0.7)", zorder=3)
ax.set_xlabel("Electricity cost ($)"); ax.set_ylabel("CO₂ (kg)")
ax.set_title("Operator's dial: trade carbon against cost", fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig("showcase/fig4_carbon_cost_tradeoff.png", dpi=170); plt.close()
print("done")
