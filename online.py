"""Rolling-horizon (online) scheduler with preemption.

Every `tick` slots the scheduler:
  1. revokes plans that haven't started yet,
  2. re-solves a CP-SAT model for all arrived-but-unfinished jobs,
  3. commits only the starts that fall inside the next tick.
In `aware` mode it learns about a grid spike when it happens and may PAUSE running
batch jobs (paying a checkpoint overhead) and resume them later or in another region.
Urgent jobs are never preempted.
"""
from dataclasses import dataclass, field
import numpy as np
from ortools.sat.python import cp_model
from data import SLOT_HOURS, HORIZON, HOME_REGION, REGIONS
from schedulers import CAPACITY_KW, MIGRATION_OVERHEAD

CHECKPOINT_SLOTS = 1        # extra slot of work lost/re-warmed on every preemption
LATE_GRACE = 6              # solver may miss a deadline by up to this many slots (heavily penalised)
SCALE = 10_000
LATE_PENALTY = 50_000_000


@dataclass
class State:
    job: object
    need: int                                   # work slots required by the next segment
    segs: list = field(default_factory=list)    # [region, start, dur]
    final: bool = False                         # last segment runs to completion
    restarts: int = 0


def _score(belief, price, forecast, alpha):
    return alpha * belief / forecast.mean() + (1 - alpha) * price / price.mean()


def _cost(job, r, s, dur, score):
    return job.power_kw * SLOT_HOURS * score[r, s:s + dur].sum() * (1 + MIGRATION_OVERHEAD * (r != HOME_REGION))


def run_online(jobs, forecast, actual, price, alpha=0.7, spike=None, aware=False,
               tick=1, time_limit=3, workers=8, verbose=False, preempt=True):
    """spike = (region, start_slot, end_slot). `aware` => scheduler observes it from start_slot."""
    states = {j.id: State(j, j.duration) for j in jobs}
    nR = len(REGIONS)
    t = 0
    while t < HORIZON and not all(s.final for s in states.values()):
        belief = forecast.copy()
        if aware and spike and t >= spike[1]:
            r0, a, b = spike
            belief[r0, max(t, a):b] = actual[r0, max(t, a):b]
        score = _score(belief, price, forecast, alpha)

        # 1. revoke plans that haven't begun
        for st in states.values():
            if st.segs and st.segs[-1][1] >= t:
                st.segs.pop()
                st.final = False

        fixed = np.zeros((nR, HORIZON))
        pending, running_pre = [], []
        for st in states.values():
            j = st.job
            if st.final and st.segs and st.segs[-1][1] + st.segs[-1][2] <= t:
                continue                                              # finished
            if st.segs and st.final and st.segs[-1][1] < t < st.segs[-1][1] + st.segs[-1][2]:
                r, s, d = st.segs[-1]
                if aware and preempt and j.tier == "batch":
                    running_pre.append(st)
                else:
                    fixed[r, t:s + d] += j.power_kw                   # non-preemptible: occupies capacity
                continue
            if j.arrival <= t:
                pending.append(st)

        model = cp_model.CpModel()
        opts = {}                         # job id -> list of (var, kind, r, s, dur)
        cap = {}
        obj = []

        def add(st, kind, r, s, dur):
            j = st.job
            v = model.NewBoolVar(f"{j.id}_{kind}_{r}_{s}")
            c = int(round(_cost(j, r, s, dur, score) * SCALE))
            late = max(0, s + dur - j.deadline)
            obj.append((c + LATE_PENALTY * late) * v)
            for tt in range(s, s + dur):
                cap.setdefault((r, tt), []).append(j.power_kw * v)
            opts.setdefault(j.id, []).append((v, kind, r, s, dur))

        def starts(j, base, dur):
            hi = min(HORIZON - dur, max(j.deadline - dur + LATE_GRACE, base + LATE_GRACE))
            return range(base, hi + 1)

        for st in pending:
            j = st.job
            for r in range(nR):
                for s in starts(j, max(j.arrival, t), st.need):
                    add(st, "start", r, s, st.need)

        for st in running_pre:
            r_cur, s0, d0 = st.segs[-1]
            rem = s0 + d0 - t
            add(st, "cont", r_cur, t, rem)
            for r in range(nR):
                for s in starts(st.job, t, rem + CHECKPOINT_SLOTS):
                    if not (r == r_cur and s == t):
                        add(st, "restart", r, s, rem + CHECKPOINT_SLOTS)

        for jid, ol in opts.items():
            model.AddExactlyOne([o[0] for o in ol])
        for (r, tt), terms in cap.items():
            model.Add(sum(terms) <= max(0, CAPACITY_KW[r] - int(fixed[r, tt])))
        if obj:
            model.Minimize(sum(obj))
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit
            solver.parameters.num_workers = workers
            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                for jid, ol in opts.items():
                    st = states[jid]
                    for v, kind, r, s, dur in ol:
                        if not solver.Value(v):
                            continue
                        if kind == "cont":
                            break
                        if kind == "restart":
                            seg = st.segs[-1]
                            seg[2] = t - seg[1]                        # truncate executed part
                            st.need, st.final = dur, False
                            st.restarts += 1
                            if verbose:
                                print(f"  t={t}: PREEMPT job {jid} ({REGIONS[seg[0]]}) -> {REGIONS[r]} @ {s}")
                        if s < t + tick:
                            st.segs.append([r, s, dur])
                            st.final = True
                        break
            elif verbose:
                print(f"  t={t}: solver status {solver.StatusName(status)}")
        t += tick
    return states


def evaluate_online(states, actual, price):
    co2 = cost = 0.0
    viol = unsched = pre = 0
    for st in states.values():
        j = st.job
        if not st.final:
            unsched += 1
            continue
        pre += st.restarts
        for r, s, d in st.segs:
            eff = 1 + MIGRATION_OVERHEAD * (r != HOME_REGION)
            kwh = j.power_kw * SLOT_HOURS
            for tt in range(s, s + d):
                co2 += kwh * actual[r, tt] * eff / 1000
                cost += kwh * price[r, tt] * eff
        last = st.segs[-1]
        viol += (last[1] + last[2]) > j.deadline
    return {"co2_kg": co2, "cost_usd": cost, "sla_violations": viol,
            "unscheduled": unsched, "preemptions": pre}


def load_profile(segs_by_job, jobs, region):
    """segs_by_job: {job_id: [(r, s, d), ...]} -> kW per slot in `region`."""
    p = {j.id: j.power_kw for j in jobs}
    load = np.zeros(HORIZON)
    for jid, segs in segs_by_job.items():
        for r, s, d in segs:
            if r == region:
                load[s:s + d] += p[jid]
    return load
