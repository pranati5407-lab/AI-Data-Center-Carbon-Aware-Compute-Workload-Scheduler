"""Carbon-aware scheduler dashboard.  Run:  streamlit run app.py"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from data import make_grid, make_jobs, REGIONS, HORIZON
from schedulers import fifo, greedy, milp, evaluate
from online import run_online, evaluate_online

st.set_page_config(page_title="Carbon-Aware Scheduler", layout="wide")
st.title("Carbon-Aware Workload Scheduler")
st.caption("Synthetic grid data and jobs (see README assumptions). FIFO runs every job on arrival in US-West.")

# ---------------- sidebar controls ----------------
with st.sidebar:
    st.header("Scenario")
    n_jobs = st.slider("Number of jobs", 20, 120, 80, 10)
    seed = st.number_input("Random seed", 0, 999, 11)
    alpha = st.slider("α  (0 = minimise cost, 1 = minimise carbon)", 0.0, 1.0, 0.7, 0.05)
    st.header("Grid spike event")
    spike_on = st.checkbox("Inject unforecast spike", value=True)
    sp_region = st.selectbox("Region", REGIONS, index=0)
    sp_start = st.slider("Start slot (30 min each)", 0, 70, 24)
    sp_len = st.slider("Duration (slots)", 2, 24, 12)
    sp_add = st.slider("Extra gCO₂/kWh", 100, 800, 450, 50)
    st.header("Schedulers")
    use_online = st.checkbox("Include online re-planning (slower, ~1 min)", value=False)
    run = st.button("Run simulation", type="primary")


@st.cache_data(show_spinner=False)
def simulate(n_jobs, seed, alpha, spike_on, sp_region, sp_start, sp_len, sp_add, use_online):
    forecast, price = make_grid()
    jobs = make_jobs(n=n_jobs, seed=seed)
    dur = {j.id: j.duration for j in jobs}
    actual = forecast.copy()
    spike = None
    if spike_on:
        r0 = REGIONS.index(sp_region)
        end = min(HORIZON, sp_start + sp_len)
        actual[r0, sp_start:end] += sp_add
        spike = (r0, sp_start, end)

    out, segs = {}, {}

    def to_segs(s):
        return {j: [(r, st_, dur[j])] for j, (r, st_) in s.items()}

    t0 = time.time(); s = fifo(jobs, forecast, price, alpha)
    out["FIFO"] = {**evaluate(jobs, s, actual, price), "preemptions": 0, "secs": time.time() - t0}
    segs["FIFO"] = to_segs(s)

    t0 = time.time(); s = greedy(jobs, forecast, price, alpha)
    out["Greedy (blind to spike)"] = {**evaluate(jobs, s, actual, price), "preemptions": 0, "secs": time.time() - t0}
    segs["Greedy (blind to spike)"] = to_segs(s)

    t0 = time.time(); s, _ = milp(jobs, forecast, price, alpha, time_limit=10)
    out["Optimizer (blind to spike)"] = {**evaluate(jobs, s, actual, price), "preemptions": 0, "secs": time.time() - t0}
    segs["Optimizer (blind to spike)"] = to_segs(s)

    if use_online:
        t0 = time.time()
        states = run_online(jobs, forecast, actual, price, alpha, spike=spike, aware=bool(spike), preempt=True)
        out["Online re-planning + preemption"] = {**evaluate_online(states, actual, price), "secs": time.time() - t0}
        segs["Online re-planning + preemption"] = {j: [tuple(x) for x in v.segs] for j, v in states.items()}

    tiers = {j.id: j.tier for j in jobs}
    return out, segs, forecast, actual, tiers, spike


@st.cache_data(show_spinner=False)
def alpha_sweep(n_jobs, seed):
    forecast, price = make_grid()
    jobs = make_jobs(n=n_jobs, seed=seed)
    rows = []
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s, _ = milp(jobs, forecast, price, a, time_limit=10)
        m = evaluate(jobs, s, forecast, price)
        rows.append((a, m["co2_kg"], m["cost_usd"]))
    return pd.DataFrame(rows, columns=["alpha", "co2_kg", "cost_usd"])


if "ran" not in st.session_state:
    st.session_state.ran = False
if run:
    st.session_state.ran = True
if not st.session_state.ran:
    st.info("Set up a scenario in the sidebar and press **Run simulation**.")
    st.stop()

with st.spinner("Solving…"):
    res, segs, forecast, actual, tiers, spike = simulate(
        n_jobs, seed, alpha, spike_on, sp_region, sp_start, sp_len, sp_add, use_online)

df = pd.DataFrame(res).T
base = df.loc["FIFO"]
df["CO₂ saved vs FIFO (%)"] = 100 * (1 - df.co2_kg / base.co2_kg)
df["Cost saved vs FIFO (%)"] = 100 * (1 - df.cost_usd / base.cost_usd)

# ---------------- headline metrics ----------------
best = df["CO₂ saved vs FIFO (%)"].drop("FIFO").idxmax()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best scheduler", best)
c2.metric("CO₂ saved", f"{base.co2_kg - df.loc[best, 'co2_kg']:.0f} kg",
          f"{df.loc[best, 'CO₂ saved vs FIFO (%)']:.1f}%")
c3.metric("Cost saved", f"${base.cost_usd - df.loc[best, 'cost_usd']:.0f}",
          f"{df.loc[best, 'Cost saved vs FIFO (%)']:.1f}%")
c4.metric("SLA misses (best / FIFO)", f"{int(df.loc[best, 'sla_violations'])} / {int(base.sla_violations)}")

tab1, tab2, tab3 = st.tabs(["Comparison", "Schedule (Gantt)", "Carbon vs cost trade-off"])

with tab1:
    show = df[["co2_kg", "cost_usd", "sla_violations", "unscheduled", "preemptions",
               "CO₂ saved vs FIFO (%)", "Cost saved vs FIFO (%)", "secs"]].round(1)
    st.dataframe(show, width="stretch")
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(df.index, df.co2_kg, color=["#888"] + ["#2a7"] * (len(df) - 1)); ax.invert_yaxis()
    ax.set_xlabel("Total CO₂ under actual grid (kg)")
    left.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for r, name in enumerate(REGIONS):
        ax.plot(actual[r], label=name)
    if spike:
        ax.axvspan(spike[1], spike[2], color="red", alpha=0.15, label="spike")
    ax.set_xlabel("slot (30 min)"); ax.set_ylabel("gCO₂/kWh"); ax.legend(fontsize=7)
    right.pyplot(fig)
    st.caption("Optimizer / greedy plans are made from the forecast and are blind to the spike. "
               "Only the online scheduler reacts to it.")

with tab2:
    which = st.selectbox("Scheduler", list(segs.keys()), index=len(segs) - 1)
    colors = ["#e07a1f", "#2a7", "#46b"]
    fig, ax = plt.subplots(figsize=(11, 6))
    order = sorted(segs[which].items(), key=lambda kv: (kv[1][0][0], kv[1][0][1]))
    for y, (jid, sg) in enumerate(order):
        for r, s, d in sg:
            ax.barh(y, d, left=s, color=colors[r], edgecolor="k" if tiers[jid] == "urgent" else "none", height=0.8)
    if spike:
        ax.axvspan(spike[1], spike[2], color="red", alpha=0.12)
    ax.set_xlabel("slot (30 min)"); ax.set_ylabel("job (sorted by region, start)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(3)], labels=REGIONS, loc="upper right")
    st.pyplot(fig)
    st.caption("Black outline = urgent job (tight deadline). Red band = grid spike window.")

with tab3:
    st.write("Sweeps α with the optimizer on the **forecast** grid (no spike).")
    if st.button("Compute trade-off curve"):
        sweep = alpha_sweep(n_jobs, seed)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sweep.cost_usd, sweep.co2_kg, "o-")
        for a, c, p in sweep.itertuples(index=False):
            ax.annotate(f"α={a}", (p, c), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Electricity cost ($)"); ax.set_ylabel("CO₂ (kg)")
        st.pyplot(fig)
        st.dataframe(sweep.round(1), width="stretch")
