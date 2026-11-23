# Carbon-Aware Workload Scheduler

Schedules flexible compute jobs across three data-center regions to cut CO₂ (and optionally cost)
without missing deadlines. Compares a FIFO baseline, a greedy heuristic, an exact
[OR-Tools CP-SAT](https://developers.google.com/optimization) optimizer, and an **online re-planning scheduler with
preemption** that reacts to unforecast grid carbon spikes.

![Where the load goes: FIFO vs carbon-aware](showcase/fig2_load_vs_carbon.png)

## Results (synthetic data, α = 0.7)

| Scenario | Result vs FIFO |
|---|---|
| Static scheduling, mean of 10 random scenarios | **27.2% ± 4.1 less CO₂**, cost about unchanged, 0 missed deadlines (FIFO averages 11) |
| …of which time-shifting alone (stay in home region) | 12.6% ± 3.5, so moving jobs *between regions* roughly doubles the saving |
| Unforecast 6 h grid spike | adaptive re-planning **39.0%** less CO₂ vs 29.5% for a plan blind to the spike |
| Long (4–10 h) jobs + 10 h spike | preemption lifts savings from 33.3% to **40.9%** |

Honest takeaways: a simple greedy heuristic gets ~97% of the optimizer's saving (26.3% vs 27.2%; the solver's proven gap to optimal is ~0.3%);
preemption barely matters for short jobs but matters a lot for long ones.

![Headline savings](showcase/fig1_headline_savings.png)

## Quick start
    pip install -r requirements.txt
    pytest -q                    # invariant tests (capacity, deadlines, preemption bookkeeping)
    python run_experiment.py     # FIFO vs greedy vs optimizer + alpha sweep  -> results.png
    python run_multiseed.py      # 10-seed robustness check (~2 min)
    python run_spike.py          # grid spike, online re-planning, preemption (~3 min)
    python run_long_jobs.py      # when does preemption pay off? (~3 min)
    streamlit run app.py         # interactive dashboard (alpha slider, spike injector, Gantt)
    uvicorn api:app --reload     # REST API, docs at http://localhost:8000/docs
    python make_showcase.py      # regenerate figures in showcase/

Real carbon data (UK regions): `python fetch_real_carbon.py && python run_experiment.py carbon_real.csv`
(the fetcher follows the API's documented format but has not been run against the live service by the author).

## How it works
Binary y[j,r,s] = 1 if job j starts in region r at slot s (30-minute slots, 48 h horizon).
- each job starts exactly once and runs contiguously; start ≥ arrival; finish ≤ deadline (SLA)
- per-region power capacity at every slot
- minimise Σ power · Δt · (α·carbon_norm + (1−α)·price_norm), +5% overhead outside the home region
- the online scheduler re-solves every 30 min and may pause running batch jobs (1 extra slot of work) and resume them later or elsewhere

## Files
| File | Purpose |
|---|---|
| `data.py` | synthetic carbon/price curves, job generator, CSV loader |
| `tests/` | pytest invariants for all schedulers and the online loop |
| `schedulers.py` | FIFO, greedy, time-shift-only greedy, CP-SAT optimizer, metrics |
| `online.py` | rolling-horizon scheduler with preemption |
| `app.py` / `api.py` | Streamlit dashboard / FastAPI wrapper |
| `run_*.py`, `make_showcase.py` | experiments and figures |
| `REPORT.md` | full report, limitations, FAQ |

## Limitations
- FIFO ignores carbon by design, so the headline saving is against a naive baseline; the time-shift-only row above is the fairer comparison for the *spatial* part.
- The 8 s/30 s CP-SAT time limits return `FEASIBLE` (not proven `OPTIMAL`) solutions, with a measured gap of ~0.3%.
- Carbon, price and job data are **synthetic**; absolute savings are illustrative, not a claim about real data centers.
- Job power draw is sampled (public cluster traces have no power data).
- The spike and long-job experiments use one scenario each; the online scheduler is told a spike's duration when it starts.
- Real UK regions share one timezone, so they offer less spatial arbitrage than the synthetic grid.
See `REPORT.md` for details.

## Data attribution
Optional real data: [Carbon Intensity API](https://carbonintensity.org.uk/) (National Energy System Operator), CC BY 4.0.

## License
MIT, see `LICENSE`.
