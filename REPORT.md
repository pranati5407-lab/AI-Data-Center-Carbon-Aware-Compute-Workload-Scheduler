# Carbon-Aware Workload Scheduling for Data Centers
*Project report*

## 1. Problem
Data centers draw electricity whose carbon intensity (gCO₂/kWh) and price vary by region and
hour. Many jobs (training runs, batch analytics, transcoding) are flexible: they can start later
or run in a cleaner region as long as they meet their deadline (SLA). The goal is to choose
**where and when** each job runs to minimise a blend of carbon and cost, with zero SLA misses
and within each region's power capacity.

## 2. Method
**Model (optimiser, OR-Tools CP-SAT).** Binary y[j,r,s] = 1 if job j starts in region r at slot s.
- Each job starts exactly once, so it runs contiguously (the start-time formulation fixes the
  "job sliced into random pieces" flaw of a plain Σx = D_j model).
- Start ≥ arrival; finish ≤ deadline (SLA).
- For every region r and slot t: Σ power of active jobs ≤ K_r (capacity 220 kW).
- Objective: min Σ P_j·Δt·(α·carbon_norm + (1−α)·price_norm), +5% overhead outside the home region.
  α is the operator's carbon-versus-cost dial.

**Baselines.** FIFO (run on arrival in the home region), a greedy heuristic (least-slack-first,
cheapest feasible slot across all regions), and a **time-shift-only greedy** (same heuristic restricted to the home region).
Greedy is the honest benchmark for whether the optimiser is worth its cost; time-shift-only separates the benefit of
moving jobs in *time* from moving them between *regions*. FIFO ignores carbon, so it is a deliberately naive reference.

**Online scheduler with preemption.** Every 30 simulated minutes it re-solves for all arrived,
unfinished jobs and commits only the next slot's starts. When it learns of a grid spike it can pause
running *batch* jobs (cost: 1 extra slot of work) and resume them later or in another region.
Urgent jobs are never preempted.

**Data.** Synthetic carbon/price curves shaped like real grids (solar dip, evening peak, wind noise,
different local-time offsets) and 80 synthetic jobs (25% urgent). `fetch_real_carbon.py` downloads
real UK regional carbon data; see Limitations.

## 3. Results (α = 0.7, 48 h horizon, 3 regions, 80 jobs)

**3a. Static scheduling**

| Scheduler | CO₂ (kg) | Cost ($) | SLA misses | CO₂ saved |
|---|---|---|---|---|
| FIFO | 1415.6 | 492.6 | 8 | n/a |
| Greedy | 980.5 | 470.4 | 0 | 30.7% |
| Optimizer | 956.7 | 472.1 | 0 | 32.4% |

Carbon-versus-cost sweep (optimizer): α = 0 gives 1681 kg and $372; α = 1 gives 938 kg and $497.
Minimising only cost emits about 79% more CO₂ than minimising only carbon, while minimising only
carbon costs about 34% more than minimising only cost.

**3a-2. Robustness: 10 random grids and job sets** (`run_multiseed.py`)

| Metric (mean ± std) | Greedy | Optimizer |
|---|---|---|
| CO₂ saved vs FIFO | 26.3% ± 3.8 | 27.2% ± 4.1 |
| Cost saved vs FIFO | -0.1% ± 3.4 | -0.2% ± 3.5 |
| SLA misses (FIFO averages 11.0) | 0 | 0 |

The optimizer beat greedy on CO₂ in 10 of 10 seeds, but by only about 0.9 points on average.
The 4% cost saving in table 3a is specific to that scenario. Across seeds, cost is roughly unchanged.
Quote the 10-seed averages as the headline; the single-scenario table is an illustration.

**Time-shift-only baseline** (same 10 seeds): greedy restricted to the home region saves **12.6% ± 3.5** CO₂ (4.9% ± 1.6 cost, 0 SLA misses).
So roughly half of the ~27% saving comes from delaying jobs into cleaner hours and the other half from moving them to cleaner
regions. The spatial half depends on the assumed spread between regions (see Limitations).

**Solver quality.** With an 8 s limit, CP-SAT returned `FEASIBLE` (not proven `OPTIMAL`) on all 10 seeds, with a final
optimality gap of 0.3% ± 0.1. "Optimal" below therefore means "within about 0.3% of the proven bound".

**3b. Unforecast spike** (US-West +450 gCO₂/kWh, 6 h, scored on the actual grid)

| Scheduler | CO₂ (kg) | SLA misses | Preemptions | CO₂ saved vs FIFO |
|---|---|---|---|---|
| FIFO | 1937 | 8 | 0 | n/a |
| Offline plan (blind) | 1366 | 0 | 0 | 29.5% |
| Online re-solve (blind) | 1356 | 0 | 0 | 30.0% |
| Online aware, no preemption | 1194 | 0 | 0 | 38.3% |
| Online aware + preemption | 1181 | 0 | 2 | 39.0% |

**3c. Long jobs** (40 jobs, batch jobs 4–10 h, 10 h spike)

| Scheduler | CO₂ (kg) | Preemptions | CO₂ saved vs FIFO |
|---|---|---|---|
| FIFO | 3007 | 0 | n/a |
| Online (blind) | 2037 | 0 | 32.2% |
| Online aware, no preemption | 2004 | 0 | 33.3% |
| Online aware + preemption | 1778 | 7 | 40.9% |

## 4. Key findings
1. Carbon-aware scheduling cuts CO₂ by about 27% ± 4 (10-seed mean) against FIFO with zero SLA misses, at roughly unchanged cost.
   About 13 points of that come from time-shifting alone; the rest comes from choosing cleaner regions.
2. **Greedy captures about 97% of the exact optimiser's saving** (26.3% vs 27.2% over 10 seeds). The optimiser's value is
   a proven reference and handling harder constraints, not a large improvement over greedy.
3. With short jobs, **reacting to the spike is worth about 8 points** (30.0% to 38.3%), but **preemption adds only
   about 1 point**. Short jobs finish before moving them pays off.
4. With long jobs the picture flips: **preemption adds about 7.6 points** (33.3% to 40.9%), because in-flight
   multi-hour jobs cannot otherwise escape a spike. Preemption matters when jobs are long relative to the spike.
5. Scheduling granularity matters: hourly ticks caused urgent-job SLA misses in testing; 30-minute ticks fixed it.

Presentation figures are in `showcase/` (regenerate with `python make_showcase.py`).

## 5. Limitations (state these up front)
- FIFO is a naive baseline (it ignores carbon and deadlines); use the time-shift-only row for a fairer view of the spatial benefit.
- Solver time limits mean results are near-optimal (gap ~0.3%), not proven optimal.
- Carbon, price and job data are synthetic; absolute savings are not a claim about real data centers.
  Savings depend on the assumed spread between regions.
- Job power draw is sampled; public cluster traces (Alibaba, Borg) have no power measurements.
- Real UK regional data shares one timezone, so it gives less spatial arbitrage than the synthetic data, and prices stay synthetic.
  The live API call was not run from the development sandbox; the parser was tested offline against the API's documented JSON shape.
- On a spike, the online scheduler is told its duration when it begins, which is optimistic.
- No data-transfer or queueing model beyond a flat 5% overhead; capacity is a single power cap per region.
- The spike and long-job experiments use a single scenario each (the static comparison is averaged over 10 seeds).
  CP-SAT is multi-threaded, so figures vary by a few tenths of a point between runs.

## 6. Likely questions
- **Why not just use greedy?** Measured: it gets about 97% of the savings. The optimiser gives a certifiable reference and
  generalises to tighter capacity and more constraints.
- **Why start-time variables instead of Σx = D?** They enforce contiguous runs without extra constraints.
- **Is 100% SLA compliance an achievement?** No, it's a hard constraint. The meaningful comparison is FIFO's 8 misses
  from congestion in a single region.
- **How do you handle forecast error?** Rolling-horizon re-planning every 30 minutes plus preemption; evaluated by scoring plans on a different "actual" grid.
- **Does it scale?** About 1 s per re-solve at 80 jobs. Beyond a few hundred jobs, use a shorter planning window or decompose by region.
- **Where would real data change the result?** Spatial savings depend on how different regions' generation mixes are.

## 7. Testing
`pytest -q` checks capacity, release times, deadlines, one-start-per-job, preemption bookkeeping (extra work = checkpoint slots × restarts,
urgent jobs never preempted) and that the time-shift-only baseline never leaves the home region.

## 7b. Future work
Multi-seed spike and long-job experiments; real multi-timezone data (Electricity Maps or WattTime); learned forecasts with uncertainty;
preemption with explicit checkpoint sizes; network-transfer costs.
