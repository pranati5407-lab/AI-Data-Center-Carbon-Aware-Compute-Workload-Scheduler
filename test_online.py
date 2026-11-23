"""Invariants for the rolling-horizon scheduler (short horizon of jobs to keep it fast)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_grid, make_jobs, HORIZON, REGIONS
from schedulers import CAPACITY_KW
from online import run_online, evaluate_online, CHECKPOINT_SLOTS


def _run(preempt):
    f, p = make_grid(7)
    jobs = make_jobs(25, 3, batch_dur=(4, 9))
    actual = f.copy()
    actual[0, 24:36] += 450
    st = run_online(jobs, f, actual, p, 0.7, spike=(0, 24, 36), aware=True,
                    preempt=preempt, time_limit=2)
    return jobs, actual, p, st


def test_online_invariants():
    jobs, actual, p, states = _run(preempt=True)
    use = np.zeros((len(REGIONS), HORIZON))
    for st in states.values():
        j = st.job
        assert st.final, f"job {j.id} unscheduled"
        assert st.segs[0][1] >= j.arrival
        worked = sum(d for _, _, d in st.segs)
        # all work done; each restart adds exactly CHECKPOINT_SLOTS of extra work
        assert worked == j.duration + CHECKPOINT_SLOTS * st.restarts
        if j.tier == "urgent":
            assert st.restarts == 0, "urgent jobs must never be preempted"
        for (r, s, d) in st.segs:
            use[r, s:s + d] += j.power_kw
        # segments must not overlap in time
        ordered = sorted(st.segs, key=lambda x: x[1])
        for a, b in zip(ordered, ordered[1:]):
            assert a[1] + a[2] <= b[1]
    for r in range(len(REGIONS)):
        assert (use[r] <= CAPACITY_KW[r] + 1e-9).all(), "capacity exceeded"
    m = evaluate_online(states, actual, p)
    assert m["unscheduled"] == 0


def test_no_preemption_when_disabled():
    _, _, _, states = _run(preempt=False)
    assert sum(st.restarts for st in states.values()) == 0
