import glob
import numpy as np
import hist
import pandas as pd
import os
import argparse
import mplhep as hep
import matplotlib.pyplot as plt

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
    "--legSize",
    type=str,
    default="small",
    help="Legend text size (small: axis ticks size, large: axis label size, number)",
)
args = parser.parse_args()

# load b field
base = "/home/submit/david_w/ceph/alca_magfield/FieldMapVersions/versions_new/"
folder = "version_18l_170812_3_8t_sx5_v9_small_fin"
files = glob.glob(f"{base}{folder}/s??_?/v-rpz-[12]00[0-5].table")

# def parse_filename(fname):
#     # Example: s11_2/v-rpz-2002.table
#     m = re.search(r"s(\d+)_([0-9])/v-rpz-(\d+)\.table", fname)
#     if m:
#         s_num, sub_num, rpz_num = map(int, m.groups())
#         return s_num, sub_num, rpz_num
#     else:
#         return None, None, None

dataframes = []

for f in sorted(files):

    df = pd.read_csv(
        f,
        sep='\s+',
        header=None,
        usecols=[0, 1, 2, 3, 4, 5], # r, phi, z, Bx, By, Bz
        names=["r", "phi", "z", "Bx", "By", "Bz"] #, "A", "B"]
    )
    dataframes.append(df)

    # Convert to tuple of tuples for a lightweight comparison
    # df_tuples = tuple(map(tuple, df.to_numpy()))
    
    # Check if this DataFrame is already in the list
    # already_exists = dataframes.to_numpy() == df_tuples

    # if already_exists:
    #     print(f"Skip {f} as it already exists")
    #     continue

    # df["source_file"] = f  # add file name column (optional)
    # df = df.sort_values(by=["r", "phi", "z"], ascending=[True, True, True])


df = pd.concat(dataframes, ignore_index=True)
df = df.sort_values(by=["r", "phi", "z"], ascending=[True, True, True])

df["Bphi"] = -df["Bx"] * np.sin(df["phi"]) + df["By"] * np.cos(df["phi"])
df["Br"] =  df["Bx"] * np.cos(df["phi"]) + df["By"] * np.sin(df["phi"])

# convert to degree
df["phi"] = df["phi"]/(2*np.pi)*360

# the values at -15 and 345 degree are the same, remove one of them
df = df[~(abs(df["phi"] + 15.0) < 0.01)].copy()


r = np.sort(df["r"].unique())
phi = np.sort(df["phi"].unique())
z = np.sort(df["z"].unique())

# there are more duplicates from the s1 and s2 (related to same boundaries?)
df = df.drop_duplicates(subset=["r", "phi", "z"], keep="first")


duplicates = df[df.duplicated(subset=["r", "phi", "z"], keep=False)]

dr = np.diff(r)
r_edges = np.array([0, *(r[1:] - dr/2), r[-1] + dr[-1]/2])
axis_r = hist.axis.Variable(r_edges, name="r", label="radius [m]", underflow=False, overflow=False)

dz = np.diff(z)
if np.allclose(dz, dz[0], rtol=1e-6):
    axis_z = hist.axis.Regular(len(z), z[0]-dz[0]/2, z[-1]+dz[0]/2, name="z", label="z [m]", underflow=False, overflow=False)
else:
    # axis_z = hist.axis.Variable(z, name="phi", label="phi [rad]", circular=True)
    raise NotImplementedError()

dphi = np.diff(phi)
if np.allclose(dphi, dphi[0], rtol=1e-6):
    axis_phi = hist.axis.Regular(len(phi), phi[0]-dphi[0]/2, phi[-1]+dphi[0]/2, name="phi", label="phi [rad]", circular=True)
else:
    raise NotImplementedError()
    # axis_phi = hist.axis.Variable(phi, name="phi", label="phi [rad]", circular=True)


axes = [axis_r, axis_phi, axis_z]

hx = hist.Hist(*axes, storage=hist.storage.Double())
hy = hist.Hist(*axes, storage=hist.storage.Double())
hz = hist.Hist(*axes, storage=hist.storage.Double())
hr = hist.Hist(*axes, storage=hist.storage.Double())
hphi = hist.Hist(*axes, storage=hist.storage.Double())

hx.values()[...] = np.reshape(df["Bx"], hx.shape)
hy.values()[...] = np.reshape(df["By"], hx.shape)
hz.values()[...] = np.reshape(df["Bz"], hx.shape)
hr.values()[...] = np.reshape(df["Br"], hx.shape)
hphi.values()[...] = np.reshape(df["Bphi"], hx.shape)


r_edges = axis_r.edges
z_edges = axis_z.edges
phi_edges = axis_phi.edges
phi_edges_rad = np.deg2rad(phi_edges)

### slizes in phi
# make histogram with phi in +/- half with +/- r
r_edges_double = np.array([-r[-1] - dr[-1]/2, *(-r[1:] + dr/2)[::-1] ,0, *(r[1:] - dr/2), r[-1] + dr[-1]/2])
axis_r_double = hist.axis.Variable(r_edges_double, name="r", label="radius [m]", underflow=False, overflow=False)
hrdouble = hist.Hist(axis_r_double, axis_z, storage=hist.storage.Double())

def make_plot(h, key, suffix, outdir):
    label = "$B_{\phi}$ in T" if key=="phi" else f"$B_{key}$ in T"
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.set_xlabel("z in m")
    ax.set_ylabel("r in m")

    plt.pcolormesh(z_edges, r_edges_double, h, cmap='plasma', shading='auto')
    plt.colorbar(label=label)

    xlim = ax.get_xlim()

    plt.plot(xlim, [0,0], color="black", linestyle="-")

    plot_tools.add_decor(
        ax,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,  # if args.dataName == "Data" and not args.noData else None,
        loc=args.titlePos,
        text_size=args.legSize,
        no_energy=True
    )

    outfile = f"field_B{key}_{suffix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    
    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={

        },
        args=args,
    )

### slizes in z
r_grid, phi_grid = np.meshgrid(r_edges, phi_edges_rad, indexing='ij')

def make_polar_plot(h, key, suffix, outdir):
    label = "$B_{\phi}$ in T" if key=="phi" else f"$B_{key}$ in T"

    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6))
    pcm = ax.pcolormesh(phi_grid, r_grid, h, cmap='plasma', shading='auto')
    fig.colorbar(pcm, ax=ax, label=label)

    ax.set_theta_zero_location("N")  # optional: 0 deg at top
    ax.set_theta_direction(-1)  # optional: clockwise

    # ax.set_xlabel("r in m")
    # ax.set_ylabel("phi in rad")

    plot_tools.add_decor(
        ax,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,  # if args.dataName == "Data" and not args.noData else None,
        loc=args.titlePos,
        text_size=args.legSize,
        no_energy=True
    )

    outfile = f"field_B{key}_{suffix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    
    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={

        },
        args=args,
    )

for h, k in (
    (hx, "x"), 
    (hy,"y"), 
    (hz,"z"), 
    (hr,"r"), 
    (hphi,"phi")
):
    outpath = f"{args.outpath}/phi_slices/B{k}"
    outdir = output_tools.make_plot_dir(outpath)

    print(f"Make B{k} plots in slizes of phi")
    nphi = len(axis_phi)//2
    for ibin in range(nphi):
        upper = h[{"phi": ibin}].values()
        lower = h[{"phi": ibin+nphi}].values()[::-1, :]

        hrdouble.values()[...] = np.concat([lower, upper])

        make_plot(hrdouble, key=k, suffix=f'phi{ibin}', outdir=outdir)

    outpath = f"{args.outpath}/z_slices/B{k}"
    outdir = output_tools.make_plot_dir(outpath)

    print(f"Make B{k} plots in slizes of z")
    for ibin in range(len(axis_z)):
        hi = h[{"z": ibin}]
        # suffix = f'zFrom{str(zbin[0]).replace("-","m").replace(".","p")}To{str(zbin[1]).replace("-","m").replace(".","p")}'
        make_polar_plot(hi, key=k, suffix=f'z{ibin}', outdir=outdir)

