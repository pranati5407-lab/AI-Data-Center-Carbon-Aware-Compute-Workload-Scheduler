"""REST API around the schedulers.   Run:  uvicorn api:app --reload   then open /docs
"""
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from data import Job, make_grid, make_jobs, REGIONS, HORIZON, SLOT_HOURS
from schedulers import fifo, greedy, milp, evaluate

app = FastAPI(title="Carbon-Aware Scheduler API", version="1.0")
FORECAST, PRICE = make_grid()


class JobIn(BaseModel):
    id: int
    arrival: int = Field(ge=0, lt=HORIZON, description="earliest start slot (30 min each)")
    duration: int = Field(ge=1, le=HORIZON, description="run length in slots")
    power_kw: int = Field(gt=0)
    deadline: int = Field(gt=0, le=HORIZON, description="job must finish by this slot")
    tier: Literal["urgent", "batch"] = "batch"


class SpikeIn(BaseModel):
    region: Literal["US-West", "EU-Central", "AP-South"]
    start: int = Field(ge=0, lt=HORIZON)
    length: int = Field(ge=1, le=HORIZON)
    add_g_per_kwh: float = Field(gt=0)


class ScheduleRequest(BaseModel):
    jobs: List[JobIn]
    alpha: float = Field(0.7, ge=0, le=1, description="0 = minimise cost, 1 = minimise carbon")
    scheduler: Literal["greedy", "optimizer"] = "optimizer"
    time_limit_s: int = Field(10, ge=1, le=60)
    spike: Optional[SpikeIn] = Field(None, description="unforecast spike used only when SCORING the plan")


@app.get("/health")
def health():
    return {"status": "ok", "slots": HORIZON, "slot_hours": SLOT_HOURS, "regions": REGIONS}


@app.get("/grid")
def grid():
    """Forecast carbon intensity (gCO2/kWh) and price ($/kWh) per region per 30-min slot."""
    return {"regions": REGIONS, "carbon": FORECAST.round(1).tolist(), "price": PRICE.round(4).tolist()}


@app.get("/sample-jobs", response_model=List[JobIn])
def sample_jobs(n: int = 20, seed: int = 11):
    return [JobIn(**j.__dict__) for j in make_jobs(n=n, seed=seed)]


@app.post("/schedule")
def schedule(req: ScheduleRequest):
    ids = [j.id for j in req.jobs]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "job ids must be unique")
    jobs = [Job(**j.model_dump()) for j in req.jobs]
    for j in jobs:
        if j.arrival + j.duration > j.deadline:
            raise HTTPException(422, f"job {j.id}: arrival + duration exceeds deadline")

    actual = FORECAST.copy()
    if req.spike:
        r0 = REGIONS.index(req.spike.region)
        actual[r0, req.spike.start:req.spike.start + req.spike.length] += req.spike.add_g_per_kwh

    base = fifo(jobs, FORECAST, PRICE, req.alpha)
    try:
        if req.scheduler == "greedy":
            plan = greedy(jobs, FORECAST, PRICE, req.alpha)
        else:
            plan, _ = milp(jobs, FORECAST, PRICE, req.alpha, time_limit=req.time_limit_s)
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    m_base, m_plan = evaluate(jobs, base, actual, PRICE), evaluate(jobs, plan, actual, PRICE)
    by_id = {j.id: j for j in jobs}
    return {
        "scheduler": req.scheduler,
        "schedule": [{"job_id": jid, "region": REGIONS[r], "start_slot": s,
                      "end_slot": s + by_id[jid].duration} for jid, (r, s) in sorted(plan.items())],
        "metrics": {k: float(v) for k, v in m_plan.items()},
        "fifo_baseline": {k: float(v) for k, v in m_base.items()},
        "co2_saved_pct": round(100 * (1 - m_plan["co2_kg"] / m_base["co2_kg"]), 1),
        "cost_saved_pct": round(100 * (1 - m_plan["cost_usd"] / m_base["cost_usd"]), 1),
    }
