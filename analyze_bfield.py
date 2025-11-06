import glob
import numpy as np
import hist
import pandas as pd
import os
import argparse
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import imageio.v2 as imageio
from io import BytesIO

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
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to folder with grid files for input",
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
    default="WiP",
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
parser.add_argument(
    "--bFieldComponents",
    type=str,
    nargs="*",
    default=["z", "r", "phi"],
    choices=["x", "y", "z", "r", "phi"],
    help="B field vector components to be plotted",
)
args = parser.parse_args()


def load_grid(input):
    print("load grid")

    files = glob.glob(f"{input}/s??_?/v-rpz-[12]00[0-5].table")

    # TODO: extend to larger radii (table 8) but has different binning
    # files.extend(glob.glob(f"{input}/s??_?/v-rpz-[12]0-8.table"))

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

    df = pd.concat(dataframes, ignore_index=True)
    df = df.sort_values(by=["r", "phi", "z"], ascending=[True, True, True])

    df["Bphi"] = -df["Bx"] * np.sin(df["phi"]) + df["By"] * np.cos(df["phi"])
    df["Br"] =  df["Bx"] * np.cos(df["phi"]) + df["By"] * np.sin(df["phi"])

    # convert to degree
    df["phi"] = df["phi"]/(2*np.pi)*360


    # the values at -15 and 345 degree are the same, remove one of them
    df = df[~(abs(df["phi"] + np.deg2rad(15.0)) < 0.0001)].copy()

    # duplicates = df[df.duplicated(subset=["r", "phi", "z"], keep=False)]

    # there are more duplicates from the s1 and s2 (related to same boundaries?)
    df = df.drop_duplicates(subset=["r", "phi", "z"], keep="first")

    return df


def make_axes(df):
    print("make axes")

    r = np.sort(df["r"].unique())
    phi = np.sort(df["phi"].unique())
    z = np.sort(df["z"].unique())

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

    return axis_r, axis_phi, axis_z


def get_histograms(df, keys = ["x", "y", "z", "r", "phi"]):
    print("make histograms")

    axes = make_axes(df)

    hists = {}

    for k in keys:
        h = hist.Hist(*axes, storage=hist.storage.Double())
        h.values()[...] = np.reshape(df[f"B{k}"], h.shape)
        hists[k] = h
    
    return hists


df = load_grid(args.input)

hists = get_histograms(df, args.bFieldComponents)

axis_r = hists["r"].axes["r"]
axis_z = hists["z"].axes["z"]
axis_phi = hists["phi"].axes["phi"]

r_edges = axis_r.edges
z_edges = axis_z.edges
phi_edges = axis_phi.edges

phi_edges_rad = np.deg2rad(phi_edges)

z_centers = z_edges[:-1] + np.diff(z_edges)/2
r_centers = r_edges[:-1] + np.diff(r_edges)/2
phi_centers = phi_edges[:-1] + np.diff(phi_edges)/2

### slizes in phi
# make histogram with phi in +/- half with +/- r
r_edges_double = np.array([*(-r_edges[1:][::-1]), *r_edges])
axis_r_double = hist.axis.Variable(r_edges_double, name="r", label="radius [m]", underflow=False, overflow=False)
hrdouble = hist.Hist(axis_r_double, axis_z, storage=hist.storage.Double())

def make_fieldlines_plot(Bz, Br, zbin, outdir, zlim=None, arrows=False):
    label = "|B| in T"

    if zlim is not None:
        norm = mcolors.Normalize(vmin=zlim[0], vmax=zlim[1])
    else:
        norm=None

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()

    ax.set_xlabel("z in m")
    ax.set_ylabel("r in m")

    r_centers = r_edges_double[:-1] + np.diff(r_edges_double)/2

    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    
    if arrows:        
        plt.quiver(Z, R, Bz, Br, np.hypot(Br, Bz), cmap='seismic', scale=50)
        plt.colorbar(label=label)
    else:
        r_vals = R.flatten()
        z_vals = Z.flatten()
        Br_vals = Br.values().flatten()
        Bz_vals = Bz.values().flatten()

        # Define uniform grid
        r_uniform = np.linspace(r_vals.min(), r_vals.max(), 200)
        z_uniform = np.linspace(z_vals.min(), z_vals.max(), 200)
        R_uniform, Z_uniform = np.meshgrid(r_uniform, z_uniform, indexing='ij')

        # Interpolate to uniform grid
        Br_uniform = griddata((r_vals, z_vals), Br_vals, (R_uniform, Z_uniform), method='linear')
        Bz_uniform = griddata((r_vals, z_vals), Bz_vals, (R_uniform, Z_uniform), method='linear')

        plt.streamplot(Z_uniform, R_uniform, Bz_uniform, Br_uniform,
            # color=np.hypot(Br_uniform, Bz_uniform),
            color='k',          # black lines
            # arrowsize=0,        # disables arrows completely
            # arrowstyle='-',
            # linestyle="--", 
            # cmap='seismic', 
            density=1)
        
        Bmag = np.sqrt(Br_uniform**2 + Bz_uniform**2)

        plt.pcolormesh(Z_uniform, R_uniform, Bmag, cmap='seismic', shading='auto', norm=norm)
        plt.colorbar(label=label)

    # # plot line for beam pipe
    # xlim = ax.get_xlim()
    # plt.plot(xlim, [0,0], color="black", linestyle="-")

    # # plot line for magnetic center
    # ylim = ax.get_ylim()
    # plt.plot([0.016, 0.016], ylim, color="black", linestyle="-")

    ax.text(1.0, 1.0, f"$\phi = {round(phi_centers[zbin])}^\circ$", ha="right", va="bottom", transform=plt.gca().transAxes)

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

    if outdir is None:
        return fig
    else:

        outfile = f"field_B_phi{zbin}"

        plot_tools.save_pdf_and_png(outdir, outfile)
        plt.close(fig)

        output_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={
            },
            args=args,
        )
        
        return None


def make_plot(h, key, suffix, outdir):
    label = "$B_{\phi}$ in T" if key=="phi" else f"$B_{key}$ in T"

    if np.min(h.values()) < 0 and np.max(h.values()) > 0:
        norm = mcolors.TwoSlopeNorm(vmin=np.min(h.values()), vcenter=0, vmax=np.max(h.values()))
    else:
        norm=None

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    
    ax.set_xlabel("z in m")
    ax.set_ylabel("r in m")

    plt.pcolormesh(z_edges, r_edges_double, h, cmap='seismic', shading='auto', norm=norm)
    plt.colorbar(label=label)

    # plot line for beam pipe
    xlim = ax.get_xlim()
    plt.plot(xlim, [0,0], color="black", linestyle="-")

    # plot line for magnetic center
    ylim = ax.get_ylim()
    plt.plot([0.016, 0.016], ylim, color="black", linestyle="-")


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
    plt.close(fig)

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

    if np.min(h.values()) < 0 and np.max(h.values()) > 0:
        norm = mcolors.TwoSlopeNorm(vmin=np.min(h.values()), vcenter=0, vmax=np.max(h.values()))
    else:
        norm=None

    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6))
    pcm = ax.pcolormesh(phi_grid, r_grid, h, cmap='seismic', shading='auto', norm=norm)
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
    plt.close(fig)
    
    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={
        },
        args=args,
    )


R, P = np.meshgrid(r_centers, phi_centers, indexing='ij')

r_vals = R.flatten()
phi_vals = P.flatten()
x = r_vals * np.cos(phi_vals)
y = r_vals * np.sin(phi_vals)
# Define uniform grid
x_uniform = np.linspace(x.min(), x.max(), 200)
y_uniform = np.linspace(y.min(), y.max(), 200)
X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform, indexing='ij')


def make_polar_fieldlines_plot(Br, Bphi, zbin, outdir, zlim=None, arrows=True):
    label = "|B| in T"

    if zlim is not None:
        norm = mcolors.Normalize(vmin=zlim[0], vmax=zlim[1])
    else:
        norm=None

    Br_vals = Br.values().flatten()
    Bphi_vals = Bphi.values().flatten()

    Bx = Br_vals * np.cos(phi_vals) - Bphi_vals * np.sin(phi_vals)  # x-component
    By = Br_vals * np.sin(phi_vals) + Bphi_vals * np.cos(phi_vals)

    # Interpolate to uniform grid
    Bx_uniform = griddata((x, y), Bx, (X_uniform, Y_uniform), method='linear')
    By_uniform = griddata((x, y), By, (X_uniform, Y_uniform), method='linear')

    step = 16

    fig, ax = plt.subplots(figsize=(6,6))

    Bmag = np.sqrt(Bx_uniform**2 + By_uniform**2)

    pcm = ax.pcolormesh(X_uniform, Y_uniform, Bmag, cmap='seismic', shading='auto', norm=norm)
    fig.colorbar(pcm, ax=ax, label=label)

    if arrows:        
        plt.quiver(
            Y_uniform[::step, ::step], 
            X_uniform[::step, ::step], 
            By_uniform[::step, ::step], 
            Bx_uniform[::step, ::step], 
            color="black", 
            scale=1,
        )
    else:
        plt.streamplot(Y_uniform, X_uniform, By_uniform, Bx_uniform, color='k', density=1.0)#, arrowsize=0)

    ax.set_xlabel('x in m')
    ax.set_ylabel('y in m')

    ax.set_xlim(min(x_uniform), max(x_uniform))
    ax.set_ylim(min(y_uniform), max(y_uniform))

    ax.text(1.0, 1.0, f"z = {round(z_centers[zbin]*100)} cm", ha="right", va="bottom", transform=plt.gca().transAxes)

    plt.gca().set_aspect('equal')

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

    if outdir is None:
        return fig
    else:
        outfile = f"field_B_z{zbin}"

        plot_tools.save_pdf_and_png(outdir, outfile)
        plt.close(fig)
        
        output_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={
            },
            args=args,
        )
        return None


nphi = len(axis_phi)//2

# for k, h in hists.items():
#     outpath = f"{args.outpath}/phi_slices/B{k}"
#     outdir = output_tools.make_plot_dir(outpath)

#     print(f"Make B{k} plots in slizes of phi")
#     for ibin in range(nphi):
#         sign = -1 if k == "r" else 1
#         upper = h[{"phi": ibin}].values()
#         lower = sign*h[{"phi": ibin+nphi}].values()[::-1, :]

#         hrdouble.values()[...] = np.concat([lower, upper])

#         make_plot(hrdouble, key=k, suffix=f'phi{ibin}', outdir=outdir)


#     outpath = f"{args.outpath}/z_slices/B{k}"
#     outdir = output_tools.make_plot_dir(outpath)

#     print(f"Make B{k} plots in slizes of z")
#     for ibin in range(len(axis_z)):
#         hi = h[{"z": ibin}]
#         # suffix = f'zFrom{str(zbin[0]).replace("-","m").replace(".","p")}To{str(zbin[1]).replace("-","m").replace(".","p")}'
#         make_polar_plot(hi, key=k, suffix=f'z{ibin}', outdir=outdir)


outpath = f"{args.outpath}/z_slices/fieldlines"
outdir = output_tools.make_plot_dir(outpath)

frames = []

print(f"Make B fieldline plots in slizes of z")
mag = (hists["phi"].values()**2 + hists["r"].values()**2)**0.5
zlim = np.min(mag), np.max(mag)

for ibin in range(len(axis_z)):
    hp = hists["phi"][{"z": ibin}]
    hr = hists["r"][{"z": ibin}]

    fig = make_polar_fieldlines_plot(hr, hp, zbin=ibin, zlim=zlim, outdir=None)#outdir)

    # Convert current figure to numpy array
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    frames.append(imageio.imread(buf))
    plt.close(fig)

outfile = f"field_B"

output_tools.write_index_and_log(
    outdir,
    outfile,
    analysis_meta_info={
    },
    args=args,
)

# To simulate a delay at the end of the loop, repeat the last frame
delay_frames = 5  # number of extra frames for delay
frames.extend([frames[-1]] * delay_frames)

imageio.mimsave(f"{outdir}/{outfile}.gif", frames, duration=0.2, loop=0)
print(f" GIF saved with {len(frames)} frames")



outpath = f"{args.outpath}/phi_slices/fieldlines"
outdir = output_tools.make_plot_dir(outpath)

frames = []

print(f"Make B fieldline plots in slizes of phi")
mag = (hists["z"].values()**2 + hists["r"].values()**2)**0.5
zlim = np.min(mag), np.max(mag)
for ibin in range(nphi):
    upper = hists["z"][{"phi": ibin}].values()
    lower = hists["z"][{"phi": ibin+nphi}].values()[::-1, :]

    hrdouble.values()[...] = np.concat([lower, upper])

    h_bz = hrdouble.copy()

    upper = hists["r"][{"phi": ibin}].values()
    lower = -hists["r"][{"phi": ibin+nphi}].values()[::-1, :]

    hrdouble.values()[...] = np.concat([lower, upper])

    h_br = hrdouble.copy()

    fig = make_fieldlines_plot(h_bz, h_br, zbin=ibin, zlim=zlim, outdir=None)#outdir)

    # Convert current figure to numpy array
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    frames.append(imageio.imread(buf))
    plt.close(fig)

# To simulate a delay at the end of the loop, repeat the last frame
delay_frames = 5  # number of extra frames for delay
frames.extend([frames[-1]] * delay_frames)

outfile = f"field_B"

output_tools.write_index_and_log(
    outdir,
    outfile,
    analysis_meta_info={
    },
    args=args,
)

imageio.mimsave(f"{outdir}/{outfile}.gif", frames, duration=0.2, loop=0)
print(f" GIF saved with {len(frames)} frames")