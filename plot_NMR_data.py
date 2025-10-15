import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import mplhep as hep

hep.style.use(hep.style.ROOT)

exts = ["png", "pdf"]

skip_surface = True
x_lim = ["2008-01-01", "2019-01-01"]
x_lim = ["2008-01-01", "2026-01-01"]
x_lim = [datetime.fromisoformat(x) for x in x_lim]

df = pd.read_csv("data/NMR_tabulated.csv").fillna(0.0)

df["day"] = df["Date"].apply(lambda x: int(x[8:10]))
df["month"] = df["Date"].apply(lambda x: int(x[5:7]))
df["year"] = df["Date"].apply(lambda x: int(x[:4]))

df["Date"] = df["Date"].apply(datetime.fromisoformat)
events = {
    "2007-01-01": "Underground",
    "2014-01-01": "Position change",
    "2017-01-01": "New electronics for current measurement",
    "2021-01-01": "Calibration w/o current in magnet",
    "2021-10-20": "Teslameter exchange",
    "2022-01-01": "Back to origin Teslameter",
    }

run1_start = datetime.fromisoformat("2008-01-01")
run2_start = datetime.fromisoformat("2014-01-01")
run2_stop = datetime.fromisoformat("2019-01-01")
run3_start = datetime.fromisoformat("2022-01-01")
run3_stop = datetime.fromisoformat("2027-01-01")

df_run1 = df.loc[(df["Date"]>run1_start) & (df["Date"]<run2_start)]
df_run2 = df.loc[(df["Date"]>run2_start) & (df["Date"]<run2_stop)]

for key in ["A", "E", "C", "D"]:
    print(f"Now at probe {key}")
    y_key = f"Channel {key} [T]"
    y_err_key = f"Channel {key} err [T]"

    x = df["Date"]
    y = df[y_key]
    y_err = df[y_err_key]

    current = df["Magnet current [A]"].values

    if skip_surface:
        x = x[1:]
        y = y[1:]
        y_err = y_err[1:]
        current = current[1:]

    # error from current measurement
    current_err = 0.02
    current_rel_err = current_err / current

    # correct the current
    y *= 18164 / current
    y_err *= 18164 / current

    y_min = min(y[y>1]-y_err[y>1])
    y_max = max(y[y>1]+y_err[y>1])
    # y_range = y_max - y_min

    fig = plt.figure()
    ax = fig.add_subplot()

    # plot RMS
    for i, (start, stop) in enumerate(
        [(run1_start, run2_start), 
        (run2_start, run2_stop), 
        (run3_start, run3_stop)]
    ):
        y_run = df.loc[(df["Date"]>start) & (df["Date"]<stop), y_key].values
        y_run = y_run[y_run > 0]

        if len(y_run) == 0:
            continue

        mean = np.mean(y_run)
        std = np.std(y_run)
        print(f"Run {i+1}: mean={mean}; std={std}")
        
        ax.plot([start, stop], [mean, mean], marker="", linestyle="-", color="black")
        ax.fill_between([start,stop],[mean-std, mean-std],[mean+std, mean+std], alpha=0.25, color="black")

    ax.plot(x, y, label="Measurement", marker="o", linestyle="", color="black")
    ax.errorbar(x, y, yerr=y_err, label="Measurement err", linestyle="", color="orange")
    ax.errorbar(x, y, yerr=y * current_rel_err, label="Current err", linestyle="", color="red")

    for evt_date, evt_note in events.items():
        date = datetime.fromisoformat(evt_date)
        ax.plot([date,date], [y_min, y_max], linestyle="--", color="grey")

    ax.set_xlim(*x_lim)
    ax.set_ylim(y_min, y_max)
    # ax.set_ylim(y_min-y_range*0.1, y_max+y_range*0.1)

    # # plot individual events
    # if key in ["C", "D"]:
    #     ax.plot([], linestyle="--", color="black")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Magnet field [T]")

    for ext in exts:
        fig.savefig(f"nmr_channel_{key}.{ext}", bbox_inches="tight")

