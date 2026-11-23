"""Invariant tests: any returned schedule must respect capacity, release times and (for the
planners) deadlines. Run:  pytest -q"""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_grid, make_jobs, HORIZON, REGIONS, HOME_REGION
from schedulers import fifo, greedy, greedy_home, milp, evaluate, CAPACITY_KW

SEEDS = [1, 2, 3]


def usage(jobs, sched):
    by_id = {j.id: j for j in jobs}
    use = np.zeros((len(REGIONS), HORIZON))
    for jid, (r, s) in sched.items():
        j = by_id[jid]
        use[r, s:s + j.duration] += j.power_kw
    return use


def check_basic(jobs, sched):
    by_id = {j.id: j for j in jobs}
    assert set(sched) == set(by_id), "every job scheduled exactly once"
    for jid, (r, s) in sched.items():
        j = by_id[jid]
        assert s >= j.arrival
        assert s + j.duration <= HORIZON
    for r in range(len(REGIONS)):
        assert (usage(jobs, sched)[r] <= CAPACITY_KW[r] + 1e-9).all(), "capacity exceeded"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fn", [fifo, greedy, greedy_home])
def test_heuristics_feasible(fn, seed):
    c, p = make_grid(seed)
    jobs = make_jobs(80, seed)
    sched = fn(jobs, c, p, 0.7)
    check_basic(jobs, sched)


@pytest.mark.parametrize("seed", SEEDS)
def test_planners_meet_deadlines(seed):
    c, p = make_grid(seed)
    jobs = make_jobs(80, seed)
    for sched in (greedy(jobs, c, p, 0.7), greedy_home(jobs, c, p, 0.7)):
        assert evaluate(jobs, sched, c, p)["sla_violations"] == 0


def test_greedy_home_stays_home():
    c, p = make_grid(1)
    jobs = make_jobs(80, 1)
    assert all(r == HOME_REGION for r, _ in greedy_home(jobs, c, p, 0.7).values())


def test_milp_feasible_and_no_worse_than_greedy():
    c, p = make_grid(1)
    jobs = make_jobs(40, 1)
    sched, info = milp(jobs, c, p, 0.7, time_limit=10)
    check_basic(jobs, sched)
    assert evaluate(jobs, sched, c, p)["sla_violations"] == 0
    # the optimiser should not emit more CO2 than greedy on the objective it minimises
    g = evaluate(jobs, greedy(jobs, c, p, 0.7), c, p)
    m = evaluate(jobs, sched, c, p)
    assert m["co2_kg"] <= g["co2_kg"] * 1.02


def test_alpha_one_minimises_carbon():
    c, p = make_grid(1)
    jobs = make_jobs(40, 1)
    co2 = {a: evaluate(jobs, greedy(jobs, c, p, a), c, p)["co2_kg"] for a in (0.0, 1.0)}
    assert co2[1.0] < co2[0.0]
