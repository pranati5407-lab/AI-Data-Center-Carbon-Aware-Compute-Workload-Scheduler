"""When does preemption actually pay off? Long batch jobs (4-10 h) + a 10-hour spike."""
import time
import pandas as pd
from data import make_grid, make_jobs, REGIONS
from schedulers import fifo, evaluate
from online import run_online, evaluate_online

ALPHA = 0.7
forecast, price = make_grid()
jobs = make_jobs(n=40, seed=5, batch_dur=(8, 21))
actual = forecast.copy()
actual[0, 22:42] += 450
spike = (0, 22, 42)

rows = []
s = fifo(jobs, forecast, price)
rows.append({"scheduler": "FIFO", **evaluate(jobs, s, actual, price), "preemptions": 0, "secs": 0})
for name, aware, pre in [("Online blind", False, False), ("Online aware, no preemption", True, False),
                         ("Online aware + preemption", True, True)]:
    t0 = time.time()
    st = run_online(jobs, forecast, actual, price, ALPHA, spike=spike, aware=aware, preempt=pre)
    rows.append({"scheduler": name, **evaluate_online(st, actual, price), "secs": round(time.time() - t0, 1)})
df = pd.DataFrame(rows).set_index("scheduler")
df["co2_vs_FIFO_%"] = (100 * (1 - df.co2_kg / df.loc["FIFO", "co2_kg"])).round(1)
print(df.round(1).to_string())
