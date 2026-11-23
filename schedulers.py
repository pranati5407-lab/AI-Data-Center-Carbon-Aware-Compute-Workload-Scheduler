"""Three schedulers: FIFO baseline, greedy heuristic, and OR-Tools CP-SAT (exact MILP-style).

A schedule is {job_id: (region_index, start_slot)}. Jobs run contiguously, in one
region, without migration (preemption is a later extension).
"""
import numpy as np
from ortools.sat.python import cp_model
from data import SLOT_HOURS, HORIZON, HOME_REGION, REGIONS

CAPACITY_KW = [220, 220, 220]      # per-region power cap K_r
MIGRATION_OVERHEAD = 0.05          # +5% energy/carbon penalty for running away from home region


def score_matrix(carbon, price, alpha):
    """Normalised blend of carbon and price so alpha is meaningful. Shape [R, T]."""
    return alpha * carbon / carbon.mean() + (1 - alpha) * price / price.mean()


def job_cost(job, r, s, score):
    base = job.power_kw * SLOT_HOURS * score[r, s:s + job.duration].sum()
    return base * (1 + MIGRATION_OVERHEAD * (r != HOME_REGION))


def _fits(use, r, s, job):
    return (use[r, s:s + job.duration] + job.power_kw <= CAPACITY_KW[r]).all()


def _place(use, r, s, job):
    use[r, s:s + job.duration] += job.power_kw


# ------------------------------------------------------------------ FIFO
def fifo(jobs, carbon, price, alpha=0.5):
    use = np.zeros((len(REGIONS), HORIZON))
    sched = {}
    for job in sorted(jobs, key=lambda j: (j.arrival, j.id)):
        s = job.arrival
        while s + job.duration <= HORIZON and not _fits(use, HOME_REGION, s, job):
            s += 1
        if s + job.duration <= HORIZON:
            _place(use, HOME_REGION, s, job)
            sched[job.id] = (HOME_REGION, s)
    return sched


# ------------------------------------------------------------------ Greedy
def greedy(jobs, carbon, price, alpha=0.5, regions=None):
    """regions: allowed region indices (default all). regions=[HOME_REGION] gives time-shifting only."""
    regions = list(range(len(REGIONS))) if regions is None else list(regions)
    score = score_matrix(carbon, price, alpha)
    use = np.zeros((len(REGIONS), HORIZON))
    sched = {}
    for job in sorted(jobs, key=lambda j: (j.deadline - j.duration - j.arrival, j.deadline)):
        best, best_c = None, np.inf
        for r in regions:
            for s in range(job.arrival, job.deadline - job.duration + 1):
                if _fits(use, r, s, job):
                    c = job_cost(job, r, s, score)
                    if c < best_c:
                        best, best_c = (r, s), c
        if best is None:  # no feasible slot: fall back to earliest fit (will miss deadline)
            for s in range(job.arrival, HORIZON - job.duration + 1):
                if _fits(use, HOME_REGION, s, job):
                    best = (HOME_REGION, s)
                    break
        if best is not None:
            _place(use, *best, job)
            sched[job.id] = best
    return sched


def greedy_home(jobs, carbon, price, alpha=0.5):
    """Stronger baseline: carbon-aware but time-shifting ONLY (stays in the home region).
    Separates the benefit of moving jobs in time from moving them between regions."""
    return greedy(jobs, carbon, price, alpha, regions=[HOME_REGION])


# ------------------------------------------------------------------ Exact optimiser
def milp(jobs, carbon, price, alpha=0.5, time_limit=30, workers=8):
    """Binary y[j,r,s] = 1 if job j starts at slot s in region r.
    Contiguity is built in (start-time formulation), unlike sum(x)=D_j.
    """
    score = score_matrix(carbon, price, alpha)
    model = cp_model.CpModel()
    y, scale = {}, 10_000
    cap_terms = {(r, t): [] for r in range(len(REGIONS)) for t in range(HORIZON)}
    obj = []

    for job in jobs:
        opts = []
        for r in range(len(REGIONS)):
            for s in range(job.arrival, job.deadline - job.duration + 1):   # release + SLA deadline
                v = model.NewBoolVar(f"y_{job.id}_{r}_{s}")
                y[job.id, r, s] = v
                opts.append(v)
                obj.append(int(round(job_cost(job, r, s, score) * scale)) * v)
                for t in range(s, s + job.duration):
                    cap_terms[r, t].append(job.power_kw * v)
        model.AddExactlyOne(opts)                                            # each job starts exactly once

    for (r, t), terms in cap_terms.items():                                  # cluster capacity
        if terms:
            model.Add(sum(terms) <= CAPACITY_KW[r])

    model.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("MILP infeasible: relax deadlines or raise capacity")

    sched = {jid: (r, s) for (jid, r, s), v in y.items() if solver.Value(v)}
    info = {"status": solver.StatusName(status), "gap": _gap(solver)}
    return sched, info


def _gap(solver):
    ub, lb = solver.ObjectiveValue(), solver.BestObjectiveBound()
    return abs(ub - lb) / max(abs(ub), 1e-9)


# ------------------------------------------------------------------ Metrics
def evaluate(jobs, sched, carbon, price):
    by_id = {j.id: j for j in jobs}
    co2_kg = cost = 0.0
    violations = unscheduled = 0
    for jid, job in by_id.items():
        if jid not in sched:
            unscheduled += 1
            continue
        r, s = sched[jid]
        kwh = job.power_kw * SLOT_HOURS
        eff = 1 + MIGRATION_OVERHEAD * (r != HOME_REGION)
        for t in range(s, s + job.duration):
            co2_kg += kwh * carbon[r, t] * eff / 1000
            cost += kwh * price[r, t] * eff
        violations += (s + job.duration > job.deadline) or (s < job.arrival)
    return {"co2_kg": co2_kg, "cost_usd": cost,
            "sla_violations": violations, "unscheduled": unscheduled}
