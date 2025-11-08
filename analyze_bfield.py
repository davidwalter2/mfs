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
from wums import boostHistHelpers as hh

hep.style.use(hep.style.ROOT)

def parseArgs():
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
        "--fieldComponents",
        type=str,
        nargs="*",
        default=["Bz", "Br", "Bphi", "Phi"],
        choices=["Bx", "By", "Bz", "Br", "Bphi", "Phi"],
        help="B field vector components or scalar field to be plotted",
    )
    parser.add_argument(
        "--makeGIFs", action="store_true", help="Make GIF plot instead of slices"
    )
    args = parser.parse_args()

    return args

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
            usecols=[0, 1, 2, 3, 4, 5, 7], # r, phi, z, Bx, By, Bz, Phi
            names=["r", "phi", "z", "Bx", "By", "Bz", "Phi"] #, "A", "B"]
        )
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df.sort_values(by=["r", "phi", "z"], ascending=[True, True, True])

    # add the earth magnetic field of strength |B| = 47.842 µT
    #   It goes from south to north and downwards on the northern hemisphere,
    #   In CMS coordinates this is in -x direction and in the -y direction
    df["Bx"] = df["Bx"] - 0.000015
    df["By"] = df["By"] - 0.000042
    df["Bz"] = df["Bz"] + 0.000015

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


def get_histograms(df, axes, keys = ["Bx", "By", "Bz", "Br", "Bphi", "Phi"]):
    print("make histograms")

    hists = {}

    for k in keys:
        h = hist.Hist(*axes, storage=hist.storage.Double())
        h.values()[...] = np.reshape(df[k], h.shape)
        hists[k] = h
    
    return hists

class Frames:
    """Utility class to collect matplotlib figures and export as GIF."""
    
    def __init__(self):
        self._frames = []

    def add_figure(self, fig, dpi=150):
        """Convert a matplotlib figure to an image frame and store it."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        self._frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    def extend_last(self, n=5):
        """Repeat the last frame n times to simulate delay at end."""
        if self._frames:
            self._frames.extend([self._frames[-1]] * n)

    def save_gif(self, outdir, outfile, duration=0.2, loop=0, args=None):
        """Save all collected frames as a GIF."""

        # To simulate a delay at the end of the loop, repeat the last frame
        self.extend_last(5)

        path = f"{outdir}/{outfile}.gif"
        imageio.mimsave(path, self._frames, duration=duration, loop=loop)
        print(f"GIF saved: {path} ({len(self._frames)} frames)")

        output_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={
            },
            args=args,
        )

    def __len__(self):
        return len(self._frames)


def get_label(key, unit="T"):
    if key=="Bphi":
        return "$B_{\phi}$ in " + unit
    elif key=="Phi":
        return "$\Phi$ in Tm"
    else:
        return f"$B_{key[1:]}$ in " + unit


def get_norm(h):
    if np.min(h.values()) < 0 and np.max(h.values()) > 0:
        return mcolors.TwoSlopeNorm(vmin=np.min(h.values()), vcenter=0, vmax=np.max(h.values()))
    else:
        return None


def make_fieldlines_plot(args, Bz, Br, r_centers, phi_centers, z_centers, phibin, outdir, zlim=None, arrows=False):
    if max(abs(Br.values().min()), abs(Br.values().max())) < 0.01 and max(abs(Bz.values().min()), abs(Bz.values().max())) < 0.01:
        label = "|B| in mT"
        scale = 1000
        Bz = hh.scaleHist(Bz, scale)
        Br = hh.scaleHist(Br, scale)
    else:
        label = "|B| in T"
        scale = 1

    if zlim is not None:
        norm = mcolors.Normalize(vmin=zlim[0]*scale, vmax=zlim[1]*scale)
    else:
        norm=None

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()

    ax.set_xlabel("z in m")
    ax.set_ylabel("r in m")

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

    ax.text(1.0, 1.0, f"$\phi = {round(phi_centers[phibin])}^\circ$", ha="right", va="bottom", transform=plt.gca().transAxes)

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

        outfile = f"field_B_phi{phibin}"

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


def make_plot(args, h, z_edges, r_edges, key, phibin, phi_centers, outdir, zlim=None):
    if max(abs(h.values().min()), abs(h.values().max())) < 0.01:
        unit="mT"
        scale = 1000
        h = hh.scaleHist(h, scale)
    else:
        unit="T"
        scale = 1

    if zlim is not None:
        norm = mcolors.Normalize(vmin=zlim[0]*scale, vmax=zlim[1]*scale)
    else:
        norm= get_norm(h)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    
    ax.set_xlabel("z in m")
    ax.set_ylabel("r in m")

    plt.pcolormesh(z_edges, r_edges, h, cmap='seismic', shading='auto', norm=norm)
    plt.colorbar(label=get_label(key, unit))

    # plot line for beam pipe
    xlim = ax.get_xlim()
    plt.plot(xlim, [0,0], color="black", linestyle="-")

    # plot line for magnetic center
    ylim = ax.get_ylim()
    plt.plot([0.016, 0.016], ylim, color="black", linestyle="-")

    ax.text(1.0, 1.0, f"$\phi = {round(phi_centers[phibin])}^\circ$", ha="right", va="bottom", transform=plt.gca().transAxes)

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
        outfile = f"field_{key}_phi{phibin}"

        plot_tools.save_pdf_and_png(outdir, outfile)

        output_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={

            },
            args=args,
        )
        return None


def make_polar_plot(args, h, phi_grid, r_grid, key, zbin, z_centers, outdir, zlim=None):
    if max(abs(h.values().min()), abs(h.values().max())) < 0.01:
        unit="mT"
        scale = 1000
        h = hh.scaleHist(h, scale)
    else:
        unit="T"
        scale = 1

    if zlim is not None:
        norm = mcolors.Normalize(vmin=zlim[0]*scale, vmax=zlim[1]*scale)
    else:
        norm= get_norm(h)

    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6))
    pcm = ax.pcolormesh(phi_grid, r_grid, h, cmap='seismic', shading='auto', norm=norm)
    fig.colorbar(pcm, ax=ax, label=get_label(key, unit))

    ax.set_theta_zero_location("N")  # optional: 0 deg at top
    ax.set_theta_direction(-1)  # optional: clockwise

    # ax.set_xlabel("r in m")
    # ax.set_ylabel("phi in rad")

    ax.text(1.0, 1.0, f"z = {round(z_centers[zbin]*100)} cm", ha="right", va="bottom", transform=plt.gca().transAxes)

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
        outfile = f"field_{key}_z{zbin}"

        plot_tools.save_pdf_and_png(outdir, outfile)
        
        output_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={
            },
            args=args,
        )
        return None

def make_polar_fieldlines_plot(args, Br, Bphi, phi_vals, x, y, x_uniform, y_uniform, X_uniform, Y_uniform, z_centers, zbin, outdir, zlim=None, arrows=True):
    if max(abs(Br.values().min()), abs(Br.values().max())) < 0.01 and max(abs(Bphi.values().min()), abs(Bphi.values().max())) < 0.01:
        label = "|B| in mT"
        scale = 1000
        Bphi = hh.scaleHist(Bphi, scale)
        Br = hh.scaleHist(Br, scale)
    else:
        label = "|B| in T"
        scale = 1

    if zlim is not None:
        norm = mcolors.Normalize(vmin=zlim[0]*scale, vmax=zlim[1]*scale)
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

def main():
    args = parseArgs()

    df = load_grid(args.input)

    axis_r, axis_phi, axis_z = make_axes(df)

    hists = get_histograms(df, [axis_r, axis_phi, axis_z], args.fieldComponents)

    r_edges = axis_r.edges
    z_edges = axis_z.edges
    phi_edges = axis_phi.edges

    phi_edges_rad = np.deg2rad(phi_edges)

    z_centers = z_edges[:-1] + np.diff(z_edges)/2
    r_centers = r_edges[:-1] + np.diff(r_edges)/2
    phi_centers = phi_edges[:-1] + np.diff(phi_edges)/2

    # make histogram with phi in +/- half with +/- r
    r_edges_double = np.array([*(-r_edges[1:][::-1]), *r_edges])
    r_centers_double = r_edges_double[:-1] + np.diff(r_edges_double)/2

    axis_r_double = hist.axis.Variable(r_edges_double, name="r", label="radius [m]", underflow=False, overflow=False)
    hrdouble = hist.Hist(axis_r_double, axis_z, storage=hist.storage.Double())

    r_grid, phi_grid = np.meshgrid(r_edges, phi_edges_rad, indexing='ij')

    R, P = np.meshgrid(r_centers, phi_centers, indexing='ij')

    r_vals = R.flatten()
    phi_vals = P.flatten()
    x = r_vals * np.cos(phi_vals)
    y = r_vals * np.sin(phi_vals)
    # Define uniform grid
    x_uniform = np.linspace(x.min(), x.max(), 200)
    y_uniform = np.linspace(y.min(), y.max(), 200)
    X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform, indexing='ij')

    if args.makeGIFs:
        outdir_base = output_tools.make_plot_dir(args.outpath)
        outdir = None

    nphi = len(axis_phi)//2

    for k, h in hists.items():

        ### slizes in phi
        if args.makeGIFs:
            frames = Frames()
            zlim = h.values().min(), h.values().max()
        else:
            zlim=None
            outpath = f"{args.outpath}/phi_slices/{k}"
            outdir = output_tools.make_plot_dir(outpath)

        print(f"Make {k} plots in slizes of phi")
        for ibin in range(nphi):
            sign = -1 if k == "Br" else 1
            upper = h[{"phi": ibin}].values()
            lower = sign*h[{"phi": ibin+nphi}].values()[::-1, :]

            hrdouble.values()[...] = np.concat([lower, upper])

            fig = make_plot(args, hrdouble, z_edges, r_edges_double, key=k, phibin=ibin, phi_centers=phi_centers, outdir=outdir, zlim=zlim)

            if args.makeGIFs:
                frames.add_figure(fig)

        if args.makeGIFs:
            frames.save_gif(outdir_base, f"field_{k}_phi_slices")

        ### slizes in z
        if args.makeGIFs:
            frames = Frames()
            zlim = h.values().min(), h.values().max()
        else:
            zlim=None
            outpath = f"{args.outpath}/z_slices/{k}"
            outdir = output_tools.make_plot_dir(outpath)

        print(f"Make {k} plots in slizes of z")
        for ibin in range(len(axis_z)):
            hi = h[{"z": ibin}]
            # suffix = f'zFrom{str(zbin[0]).replace("-","m").replace(".","p")}To{str(zbin[1]).replace("-","m").replace(".","p")}'
            fig = make_polar_plot(args, hi, phi_grid, r_grid, key=k, zbin=ibin, z_centers=z_centers, outdir=outdir, zlim=zlim)

            if args.makeGIFs:
                frames.add_figure(fig)

        if args.makeGIFs:
            frames.save_gif(outdir_base, f"field_{k}_z_slices")


    if "Bphi" in hists and "Br" in hists.keys():
        if args.makeGIFs:
            frames = Frames()
            mag = (hists["Bphi"].values()**2 + hists["Br"].values()**2)**0.5
            zlim = np.min(mag), np.max(mag)
        else:
            zlim=None
            outpath = f"{args.outpath}/z_slices/fieldlines"
            outdir = output_tools.make_plot_dir(outpath)


        print(f"Make B fieldline plots in slizes of z")
        for ibin in range(len(axis_z)):
            hp = hists["Bphi"][{"z": ibin}]
            hr = hists["Br"][{"z": ibin}]

            fig = make_polar_fieldlines_plot(args, hr, hp, phi_vals, x, y, x_uniform, y_uniform, X_uniform, Y_uniform, z_centers, zbin=ibin, zlim=zlim, outdir=outdir)
            if args.makeGIFs:
                frames.add_figure(fig)

        if args.makeGIFs:
            frames.save_gif(outdir_base, "field_z_slices")

    if "Bz" in hists and "Br" in hists.keys():
        if args.makeGIFs:
            frames = Frames()
            mag = (hists["Bz"].values()**2 + hists["Br"].values()**2)**0.5
            zlim = np.min(mag), np.max(mag)
        else:
            zlim=None
            outpath = f"{args.outpath}/phi_slices/fieldlines"
            outdir = output_tools.make_plot_dir(outpath)

        print(f"Make B fieldline plots in slizes of phi")
        for ibin in range(nphi):
            upper = hists["Bz"][{"phi": ibin}].values()
            lower = hists["Bz"][{"phi": ibin+nphi}].values()[::-1, :]

            hrdouble.values()[...] = np.concat([lower, upper])

            h_bz = hrdouble.copy()

            upper = hists["Br"][{"phi": ibin}].values()
            lower = -hists["Br"][{"phi": ibin+nphi}].values()[::-1, :]

            hrdouble.values()[...] = np.concat([lower, upper])

            h_br = hrdouble.copy()

            fig = make_fieldlines_plot(args, h_bz, h_br, r_centers_double, phi_centers, z_centers, phibin=ibin, zlim=zlim, outdir=outdir)
            
            if args.makeGIFs:
                frames.add_figure(fig)

        if args.makeGIFs:
            frames.save_gif(outdir_base, "fieldlines_phi_slices")

if __name__ == "__main__":
    main()