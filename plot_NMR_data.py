import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import mplhep as hep
import argparse
import os
import re

from wums import output_tools, plot_tools

hep.style.use(hep.style.ROOT)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=3,
    choices=[0, 1, 2, 3, 4],
    help="Set verbosity level with logging, the larger the more verbose",
)
parser.add_argument(
    "--noColorLogger", action="store_true", help="Do not use logging with colors"
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    default=os.path.expanduser("./test"),
    help="Base path for output",
)
parser.add_argument(
    "--title",
    default="CMS",
    type=str,
    help="Title to be printed in upper left",
)
parser.add_argument(
    "--subtitle",
    default="Work in progress",
    type=str,
    help="Subtitle to be printed after title",
)
parser.add_argument("--titlePos", type=int, default=0, help="title position")
parser.add_argument(
    "--legPos", type=str, default="best", help="Set legend position"
)
parser.add_argument(
    "--legSize",
    type=str,
    default="small",
    help="Legend text size (small: axis ticks size, large: axis label size, number)",
)
parser.add_argument(
    "--legCols", type=int, default=2, help="Number of columns in legend"
)
parser.add_argument(
    "--rrange",
    type=float,
    nargs=2,
    default=[0.9, 1.1],
    help="y range for ratio plot",
)
parser.add_argument(
    "--skipSurface", action="store_true", help="Skip surface measurement"
)
args = parser.parse_args()

outdir = output_tools.make_plot_dir(args.outpath)


def read_predictions(input_file):
    pattern = re.compile(r"B=\s*\(([^)]+)\)")
    B_values = {}
    keys = ["A", "E", "C", "D"]

    # Read file and extract matches
    with open(input_file, "r") as f:
        i = 0
        for line in f:
            match = pattern.search(line)
            if match:
                # Extract the numbers and split into floats
                values = np.array([float(v.strip()) for v in match.group(1).split(",")])
                B_values[keys[i]] = np.sum(values**2)**0.5 # take the magnitude
                i += 1
    return B_values

# file with predictions from CMSSW
predictions = {
    "120812": {
        "values": {
            datetime.fromisoformat("2012-01-01"): read_predictions("data/field_results_120812_run1.txt"),
            datetime.fromisoformat("2016-01-01"): read_predictions("data/field_results_120812_run2.txt")
            },
        "color":"cyan",
        "marker": "D"
        },
    "130503": {
        "values": {
            datetime.fromisoformat("2012-01-01"): read_predictions("data/field_results_130503_run1.txt"),
            datetime.fromisoformat("2016-01-01"): read_predictions("data/field_results_130503_run2.txt")
            },
        "color":"purple",
        "marker": "*"
        },
    "160812": {
        "values": {
            datetime.fromisoformat("2012-01-01"): read_predictions("data/field_results_160812_run1.txt"),
            datetime.fromisoformat("2016-01-01"): read_predictions("data/field_results_160812_run2.txt")
            },
        "color":"green",
        "marker": "X"
        },
    "170812": {
        "values": {
            datetime.fromisoformat("2006-08-28"): read_predictions("data/field_results_170812_sx5.txt"),
            datetime.fromisoformat("2012-01-01"): read_predictions("data/field_results_170812_run1.txt"),
            datetime.fromisoformat("2016-01-01"): read_predictions("data/field_results_170812_run2.txt")
            },
        "color":"blue",
        "marker": "P"
        },
    "PolyFit2D": {
        "values": {
            datetime.fromisoformat("2006-08-28"): read_predictions("data/field_results_polyfit2d.txt"),
            },
        "color":"orange",
        "marker": "^"
        },
    "PolyFit3D": {
        "values": {
            datetime.fromisoformat("2006-08-28"): read_predictions("data/field_results_polyfit3d.txt"),
            },
        "color":"red",
        "marker": "v"
        },
    }

# times to compute the ratios to the measurements
meas_times = [datetime.fromisoformat("2006-08-28"), datetime.fromisoformat("2012-01-01"), datetime.fromisoformat("2016-01-01"), datetime.fromisoformat("2024-01-01")]

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

underground_start = datetime.fromisoformat("2006-01-01")
underground_stop = datetime.fromisoformat("2007-01-01")
run1_start = datetime.fromisoformat("2010-01-01")
run2_start = datetime.fromisoformat("2014-01-01")
run2_stop = datetime.fromisoformat("2019-01-01")
run3_start = datetime.fromisoformat("2022-01-01")
run3_stop = datetime.fromisoformat("2027-01-01")

y_key = lambda key: f"Channel {key} [T]"
y_err_key = lambda key: f"Channel {key} err [T]"

def make_plot(current, y, y_err, x, df, key, x_lim, title_right=None):
    x_lim = [datetime.fromisoformat(x) for x in x_lim]

    # error from current measurement
    current_err = 0.02
    current_rel_err = current_err / current

    y_min = min(y[y!=0]-y_err[y!=0])
    y_max = max(y[y!=0]+y_err[y!=0])

    fig = plt.figure()
    ax1 = fig.add_subplot(6, 1, (1, 4))
    ax2 = fig.add_subplot(6, 1, (5, 8))

    ax2.set_xlabel("Date")
    ax1.set_ylabel("Magnet field [T]")
    ax2.set_ylabel("Ratio")

    # plot probes measured at lower current in grey
    masked_scaled = current != 18164
    if key == "A":
        y_original_err = (y_err**2 + (y * current_rel_err)**2)**0.5
        ax1.errorbar(
            x[masked_scaled], 
            y[masked_scaled], 
            yerr=y_original_err[masked_scaled], 
            marker=".", 
            capsize=5, 
            capthick=2, 
            linestyle="", 
            color="grey", 
            label="Meas. (18160A)"
            )

    # correct the current
    y *= 18164 / current
    y_err *= 18164 / current
    y_err = (y_err**2 + (y * current_rel_err)**2)**0.5

    # plot RMS
    means = {}
    mean_tmp = 0
    for i, (start, stop) in enumerate([
        (underground_start, underground_stop),
        (run1_start, run2_start),
        (run2_start, run2_stop), 
        (run3_start, run3_stop)
    ]):
        if key == "AminusE":
            y_run = np.diff(df.loc[(df["Date"]>start) & (df["Date"]<stop), [y_key("E"), y_key("A")]].values).squeeze()
        elif key == "DminusC":
            y_run = np.diff(df.loc[(df["Date"]>start) & (df["Date"]<stop), [y_key("C"), y_key("D")]].values).squeeze()
        else:
            y_run = df.loc[(df["Date"]>start) & (df["Date"]<stop), y_key(key)].values
        y_run = y_run[y_run != 0]

        if len(y_run) == 0:
            continue

        mean = np.mean(y_run)
        std = np.std(y_run)
        print(f"Run {i}: mean={mean}; std={std}")

        if mean != 0:
            relDiff = (mean/mean_tmp-1) #* 1e5
            print(f"Rel diff. = {relDiff}")
        mean_tmp = mean

        ax1.plot([start, stop], [mean, mean], marker="", linestyle="-", color="black")
        ax1.fill_between([start,stop],[mean-std, mean-std],[mean+std, mean+std], alpha=0.25, color="black")

        # lower panel
        ax2.fill_between([start,stop],[1-std, 1-std],[1+std, 1+std], alpha=0.25, color="black")

        means[meas_times[i]] = mean

    ax2.plot(x_lim, [1, 1], marker="", linestyle="-", color="black")

    # ax.plot(x, y, label="Measurement", marker="o", linestyle="", color="black")
    ax1.errorbar(x, y, yerr=y_err, label="Meas.", marker=".", capsize=5, capthick=2, linestyle="", color="black")
    # ax.errorbar(x, y, yerr=y * current_rel_err, label="Current err", linestyle="", color="red")

    for pred_label, pred_dict in predictions.items():
        pred_times = []
        pred_values = []
        for pred_time, pred_items in pred_dict["values"].items():
            if key == "AminusE":
                pred_value = pred_items["A"] - pred_items["E"]
            elif key == "DminusC":
                pred_value = pred_items["D"] - pred_items["C"]
            else:
                pred_value = pred_items[key]
            if pred_value == 0:
                continue
            pred_values.append(pred_value)
            pred_times.append(pred_time)

        if len(pred_times) > 0:
            ax1.plot(pred_times, pred_values, linewidth=0, linestyle=None, marker=pred_dict["marker"], color=pred_dict["color"], label=pred_label)

            y_min = min(y_min, min(pred_values))
            y_max = max(y_max, max(pred_values))

            meas_means = np.array([means[t] for t in pred_times if t in means.keys()])
    
            meas_pred_times = np.array([t for t in pred_times if t in means.keys()])
            meas_pred_values = np.array([v for t,v in zip(pred_times, pred_values) if t in means.keys()])

            ax2.plot(meas_pred_times, meas_pred_values/meas_means, linewidth=0, linestyle=None, marker=pred_dict["marker"], color=pred_dict["color"])

    y_range = y_max - y_min
    y_lim = [y_min - y_range*0.4, y_max + y_range * 0.1]

    for evt_date, evt_note in events.items():
        date = datetime.fromisoformat(evt_date)
        ax1.plot([date,date], [y_min, y_lim[1]], linestyle="--", color="grey", zorder=0)

    ax1.set_xlim(*x_lim)
    ax2.set_xlim(*x_lim)
    ax1.set_ylim(*y_lim)

    if args.rrange is not None:
        ax2.set_ylim(*args.rrange)

    ax1.set_xticklabels([])

    ax1.get_yaxis().get_major_formatter().set_useOffset(False)

    # ax.set_ylim(y_min-y_range*0.1, y_max+y_range*0.1)

    # # plot individual events
    # if key in ["C", "D"]:
    #     ax.plot([], linestyle="--", color="black")

    plot_tools.add_decor(
        ax1,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,  # if args.dataName == "Data" and not args.noData else None,
        loc=args.titlePos,
        text_size=args.legSize,
        no_energy=True
    )

    title_right = f"Probe {key}" if title_right is None else title_right
    ax1.text(1.0, 1.0, title_right, ha='right', va='bottom', transform=ax1.transAxes)

    plot_tools.addLegend(
        ax1,            
        ncols=args.legCols,
        loc=args.legPos,
        text_size=args.legSize,
    )

    outfile = f"nmr_channel_{key}"

    plot_tools.save_pdf_and_png(outdir, outfile)

    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={

        },
        args=args,
    )

for key in ["A", "E", "C", "D"]:
    print(f"Now at probe {key}")
    x = df["Date"]
    y = df[y_key(key)]
    y_err = df[y_err_key(key)]
    y_err[y_err < 0.00003] = 0.00003

    current = df["Magnet current [A]"].values

    if key == "A":
        if args.skipSurface:
            x_lim = ["2008-01-01", "2026-01-01"]
            x = x[1:]
            y = y[1:]
            y_err = y_err[1:]
            current = current[1:]
        else:
            x_lim = ["2006-01-01", "2026-01-01"]
    elif key == "E":
        x_lim = ["2006-01-01", "2026-01-01"]
    else:
        x_lim = ["2006-01-01", "2019-01-01"]

    make_plot(current, y, y_err, x, df, key, x_lim)


# plot difference between A and E
y_key_A = f"Channel A [T]"
y_key_E = f"Channel E [T]"

mask = (df[y_key_E] != 0) & (df[y_key_A] != 0)

df = df[mask]

x = df["Date"]

y = df[y_key("A")].values - df[y_key("E")].values
y_err = (df[y_err_key("A")].values**2 + df[y_err_key("E")].values**2)**0.5

current = df["Magnet current [A]"].values

x_lim = ["2006-01-01", "2026-01-01"]

make_plot(current, y, y_err, x, df, "AminusE", x_lim, "Probe A - Probe E")

# plot difference between D and C
y_key_C = f"Channel C [T]"
y_key_D = f"Channel D [T]"

mask = (df[y_key_D] != 0) & (df[y_key_C] != 0)

df = df[mask]

x = df["Date"]

y = df[y_key("D")].values - df[y_key("C")].values
y_err = (df[y_err_key("D")].values**2 + df[y_err_key("C")].values**2)**0.5

current = df["Magnet current [A]"].values

x_lim = ["2006-01-01", "2019-01-01"]

make_plot(current, y, y_err, x, df, "DminusC", x_lim, "Probe D - Probe C")