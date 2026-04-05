"""
Plot harmonic fit residuals: overall histograms and phase-space slices.

Usage:
  python plot_residuals.py
  python plot_residuals.py --fit tosca170812_extended_coeffs_lmax18_all3_rmax290.npz
  python plot_residuals.py --fit tosca170812_full_coeffs_lmax18_all3.npz --rmax-sphere 280
  python plot_residuals.py --fit myfit.npz --grid field.txt --rmax-sphere 290 --out residuals.pdf
"""

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os
import mplhep as hep
from wums import output_tools, plot_tools
hep.style.use(hep.style.ROOT)

# Tracker volume boundaries (from CMSSW ParametrizedEngine isDefined())
# OAE (default parametrization, standard tracking volume): r < 115 cm, |z| < 280 cm
# PolyFit3D (CVH refit validity, extended): r < 190 cm, |z| < 350 cm, |z|+2.5r < 670 cm
TRACKER_OAE_RMAX  = 115.0   # cm
TRACKER_OAE_ZMAX  = 280.0   # cm
TRACKER_P3D_RMAX  = 190.0   # cm
TRACKER_P3D_ZMAX  = 350.0   # cm  (also corner cut |z|+2.5r < 670 cm)

# CMSSW field map volume z-boundaries inside the tracker region.
# Extracted from MagneticField/Interpolation/data/grid_170812_3_8t/log_convert.txt
# (X3 / X3+(N3-1)*A3 values of rφz-volumes that begin or end within |z| < 300 cm).
# z ≈ ±126.8 cm : coil central-module boundary (module half-length 125.6 cm)
# z ≈ ±142.3 cm : vacuum vessel / thermal shield layer boundary
# z ≈ ±181.3 cm : cryogenic chimney volume boundary (added map version 1103_071212)
CMSSW_VOL_Z = [126.8, 142.3, 181.3]   # cm, positive side; mirrored to ±


def add_volume_boundaries(ax, orientation='z-axis'):
    """Draw black dashed lines at CMSSW field-map volume z-boundaries.

    orientation : 'z-axis'  — ax.axvline (z is the x-axis, as in r-z maps)
                  'z-yaxis' — ax.axhline (z is the y-axis, as in phi-z maps)
    """
    kw = dict(color='black', lw=0.8, ls='--', alpha=0.7)
    for z0 in CMSSW_VOL_Z:
        for z in [+z0, -z0]:
            if orientation == 'z-axis':
                ax.axvline(z, **kw)
            else:
                ax.axhline(z, **kw)


def add_tracker_boundaries(ax):
    """Overlay tracker boundary rectangles on an r-z axes.

    Draws two rectangles:
      - solid blue:   OAE/standard tracker  r<115 cm, |z|<280 cm
      - dashed green: PolyFit3D/CVH validity r<190 cm, |z|<350 cm
        (the diagonal corner cut |z|+2.5r<670 is not drawn for simplicity)
    """
    # OAE tracker rectangle (full z range: -zmax to +zmax)
    rect_oae = mpatches.Rectangle(
        (-TRACKER_OAE_ZMAX, 0),
        2 * TRACKER_OAE_ZMAX, TRACKER_OAE_RMAX,
        linewidth=1.5, edgecolor='blue', facecolor='none',
        linestyle='-', label='OAE tracker',
        zorder=5,
    )
    # PolyFit3D / CVH validity rectangle
    rect_p3d = mpatches.Rectangle(
        (-TRACKER_P3D_ZMAX, 0),
        2 * TRACKER_P3D_ZMAX, TRACKER_P3D_RMAX,
        linewidth=1.5, edgecolor='green', facecolor='none',
        linestyle='--', label='PolyFit3D',
        zorder=5,
    )
    ax.add_patch(rect_oae)
    ax.add_patch(rect_p3d)
    ax.legend(loc='upper right')


def _add_cms_label(fig, args):
    """Add CMS experiment label to the first axes of a figure."""
    if fig.axes:
        plot_tools.add_decor(
            fig.axes[0], args.title, args.subtitle,
            data=False, lumi=None, loc=args.titlePos, no_energy=True)


from harmonic_basis import eval_field
from cylindrical_basis import eval_field_cyl, parse_mode_label
from cylindrical_bessel_basis import eval_field_cjb, parse_mode_label as parse_mode_label_cjb
from fit_field import load_grid, vol_boundary_mask, CMSSW_VOL_BOUNDARIES_CM, parse_m_max_per_l
import zernike_basis


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_fit(npz_path):
    """Load a fit npz (spherical or cylindrical).  Returns (coeffs, params, meta).

    meta keys vary by basis:
      spherical:   r_scale, z0, l_max
      cylindrical: L, n_max, m_max
    plus: basis, components, col_norms, tikhonov, condition_number
    """
    d = np.load(npz_path, allow_pickle=True)
    coeffs = d['coeffs']
    param_strs = list(d['params'])
    basis_type = str(d.get('basis_type', d.get('basis', 'spherical')))

    def _scalar(key, default=np.nan):
        return float(d[key]) if key in d else default

    if basis_type == 'cylindrical':
        modes = [parse_mode_label(s) for s in param_strs]
        meta = {
            'basis': 'cylindrical',
            'L':     float(d['L']),
            'n_max': int(d['n_max']),
            'm_max': int(d['m_max']),
            'components':             list(d.get('components', ['Bz', 'Br', 'Bphi'])),
            'rmax_cm':                _scalar('rmax_cm'),
            'zmax_cm':                _scalar('zmax_cm'),
            'rmax_sphere_cm':         _scalar('rmax_sphere_cm'),
            'rmin_sphere_cm':         _scalar('rmin_sphere_cm'),
            'exclude_vol_boundaries': bool(d['exclude_vol_boundaries']) if 'exclude_vol_boundaries' in d else False,
        }
        return coeffs, modes, meta

    if basis_type == 'bessel':
        modes = [parse_mode_label_cjb(s) for s in param_strs]
        meta = {
            'basis': 'bessel',
            'R':     float(d['R']),
            'n_max': int(d['n_max']),
            'm_max': int(d['m_max']),
            'components':             list(d.get('components', ['Bz', 'Br', 'Bphi'])),
            'rmax_cm':                _scalar('rmax_cm'),
            'zmax_cm':                _scalar('zmax_cm'),
            'rmax_sphere_cm':         _scalar('rmax_sphere_cm'),
            'rmin_sphere_cm':         _scalar('rmin_sphere_cm'),
            'exclude_vol_boundaries': bool(d['exclude_vol_boundaries']) if 'exclude_vol_boundaries' in d else False,
        }
        return coeffs, modes, meta

    elif basis_type == 'zernike':
        params = []
        for s in param_strs:
            parts = s.split('_')
            params.append((int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]), parts[3]))
        meta = {
            'basis':                  'zernike',
            'r_scale':                float(d['r_scale']),
            'z0':                     float(d['z0']) if 'z0' in d else 0.0,
            'n_max':                  int(d['n_max']) if 'n_max' in d else -1,
            'l_max':                  int(d['l_max']) if 'l_max' in d else -1,
            'components':             list(d.get('components', ['Bz', 'Br', 'Bphi'])),
            'rmax_cm':                _scalar('rmax_cm'),
            'zmax_cm':                _scalar('zmax_cm'),
            'rmax_sphere_cm':         _scalar('rmax_sphere_cm'),
            'rmin_sphere_cm':         _scalar('rmin_sphere_cm'),
            'exclude_vol_boundaries': bool(d['exclude_vol_boundaries']) if 'exclude_vol_boundaries' in d else False,
        }
        return coeffs, params, meta

    else:  # harmonic / spherical
        params = []
        for s in param_strs:
            parts = s.split('_')
            params.append((int(parts[0][1:]), int(parts[1][1:]), parts[2]))
        l_max_phi = int(d['l_max_phi']) if 'l_max_phi' in d else None
        n_max_sum = int(d['n_max_sum']) if 'n_max_sum' in d else None
        mmpl_str  = str(d['m_max_per_l_str']) if 'm_max_per_l_str' in d else None
        m_max_per_l = (parse_m_max_per_l(mmpl_str) if mmpl_str is not None else None)
        meta = {
            'basis':                  'harmonic',
            'r_scale':                float(d['r_scale']),
            'z0':                     float(d['z0']) if 'z0' in d else 0.0,
            'l_max':                  int(d['l_max']) if 'l_max' in d else -1,
            'l_max_phi':              l_max_phi,
            'n_max_sum':              n_max_sum,
            'm_max_per_l':            m_max_per_l,
            'm_max_per_l_str':        mmpl_str,
            'components':             list(d.get('components', ['Bz', 'Br', 'Bphi'])),
            'rmax_cm':                _scalar('rmax_cm'),
            'zmax_cm':                _scalar('zmax_cm'),
            'rmax_sphere_cm':         _scalar('rmax_sphere_cm'),
            'rmin_sphere_cm':         _scalar('rmin_sphere_cm'),
            'exclude_vol_boundaries': bool(d['exclude_vol_boundaries']) if 'exclude_vol_boundaries' in d else False,
        }
        return coeffs, params, meta


def compute_residuals(grid, coeffs, params_or_modes, meta,
                      components=('Bz', 'Br', 'Bphi')):
    """Return dict of residual arrays (measured - predicted) in mT."""
    if meta['basis'] == 'bessel':
        pred = eval_field_cjb(coeffs, params_or_modes,
                              grid['r_cm'], grid['phi'], grid['z_cm'],
                              R=meta['R'], components=components)
    elif meta['basis'] == 'cylindrical':
        pred = eval_field_cyl(coeffs, params_or_modes,
                              grid['r_cm'], grid['phi'], grid['z_cm'],
                              L=meta['L'], components=components)
    elif meta['basis'] == 'zernike':
        pred = zernike_basis.eval_field(coeffs, params_or_modes,
                                        grid['r_cm'], grid['phi'], grid['z_cm'],
                                        components=components,
                                        r_scale=meta['r_scale'], z0=meta['z0'])
    else:
        pred = eval_field(coeffs, params_or_modes,
                          grid['r_cm'], grid['phi'], grid['z_cm'],
                          components=components,
                          r_scale=meta['r_scale'], z0=meta['z0'])
    residuals = {}
    for comp in components:
        residuals[comp] = (grid[comp] - pred[comp]) * 1e3   # → mT
    return residuals


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

COMP_LABELS = {'Bz': r'$B_z$', 'Br': r'$B_r$', 'Bphi': r'$B_\phi$'}
COMP_COLORS = {'Bz': '#1f77b4', 'Br': '#ff7f0e', 'Bphi': '#2ca02c'}


def make_histogram_page(residuals, components, title, ax_arr, region_label=None):
    """Fill a row of histogram axes, one per component."""
    for ax, comp in zip(ax_arr, components):
        res = residuals[comp]
        rms = np.sqrt(np.mean(res**2))
        mu  = np.mean(res)
        bins = np.linspace(-3*rms, 3*rms, 80)
        ax.hist(res, bins=bins, color=COMP_COLORS[comp], alpha=0.75, density=True)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axvline(mu, color='red', lw=1.0, ls='-')
        ax.set_xlabel('residual [mT]')
        ax.set_ylabel('density')
        # expand y-axis to give clear headroom above histogram for text
        yhi = ax.get_ylim()[1]
        ax.set_ylim(0, yhi * 1.5)
        # stats block: upper left — RMS, mean, N
        ax.text(0.02, 0.98,
                f'RMS = {rms:.4f} mT\nmean = {mu:+.4f} mT\nN = {len(res):,}',
                ha='left', va='top', transform=ax.transAxes,
                fontsize='small', linespacing=1.4)
        # component label: upper right
        ax.text(0.98, 0.98, COMP_LABELS[comp],
                ha='right', va='top', transform=ax.transAxes)
        # region label: lower right (avoids crowding the top)
        if region_label is not None:
            ax.text(0.98, 0.02, region_label,
                    ha='right', va='bottom', transform=ax.transAxes,
                    fontsize='small')


def slice_plot(ax, x_vals, res_vals, xlabel, comp, nbins=30,
               profile_only=False):
    """
    Profile mean ± RMS vs x_vals, drawn into an existing axes.
    """
    rms_global = np.sqrt(np.mean(res_vals**2))

    # bin centres and per-bin mean/rms
    edges = np.percentile(x_vals, np.linspace(0, 100, nbins+1))
    edges = np.unique(edges)
    centres, means, rmss = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x_vals >= lo) & (x_vals < hi)
        if mask.sum() < 3:
            continue
        centres.append(0.5*(lo+hi))
        means.append(np.mean(res_vals[mask]))
        rmss.append(np.sqrt(np.mean(res_vals[mask]**2)))
    centres = np.array(centres)
    means   = np.array(means)
    rmss    = np.array(rmss)

    ax.fill_between(centres, means - rmss, means + rmss,
                    alpha=0.25, color=COMP_COLORS[comp], label='±RMS')
    ax.plot(centres, means, '-o', ms=3, color=COMP_COLORS[comp], label='mean')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axhline(+rms_global, color='grey', lw=0.6, ls=':')
    ax.axhline(-rms_global, color='grey', lw=0.6, ls=':')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('residual [mT]')
    ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
    ax.legend()
    return ax


def make_slice_page(fig, residuals, grid, components, var, xlabel, nbins=30):
    """Kept for back-compat; not used in main any more."""
    n = len(components)
    gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.45)
    for i, comp in enumerate(components):
        ax = fig.add_subplot(gs[i])
        slice_plot(ax, var, residuals[comp], xlabel, comp, nbins=nbins)


def make_2d_map(ax, x, y, residuals, comp, xlabel, ylabel,
                nx=40, ny=40, vmax=None):
    """2D mean-residual map in (x,y) bins."""
    x_edges = np.linspace(x.min(), x.max(), nx+1)
    y_edges = np.linspace(y.min(), y.max(), ny+1)
    img = np.full((ny, nx), np.nan)
    for ix in range(nx):
        for iy in range(ny):
            mask = ((x >= x_edges[ix]) & (x < x_edges[ix+1]) &
                    (y >= y_edges[iy]) & (y < y_edges[iy+1]))
            if mask.sum() >= 2:
                img[iy, ix] = np.mean(residuals[mask])
    if vmax is None:
        vmax = np.nanpercentile(np.abs(img), 95)
    im = ax.imshow(img, origin='lower', aspect='auto',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label='mean residual [mT]')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fit', default='data/fitresults/tosca170812_extended_coeffs_lmax18_all3_rmax290.npz',
                        help='NPZ file with fit coefficients')
    parser.add_argument('--grid', default='../MagneticField/Engine/test/field_tosca170812_extended.txt',
                        help='Field grid file to evaluate residuals on')
    parser.add_argument('--rmax-sphere', type=float, default=np.nan,
                        help='Spherical rmax cut applied when loading the grid [cm] '
                             '(default: read from fit npz)')
    parser.add_argument('--out', default=None,
                        help='Output PDF path (default: derived from --fit name)')
    parser.add_argument('--nbins', type=int, default=35,
                        help='Number of bins for profile plots')
    parser.add_argument('--no-vol-boundaries', action='store_true',
                        help='Suppress CMSSW volume boundary lines (use for PolyFit3D/smooth models)')
    parser.add_argument('--title', default='CMS', help='Experiment label (default: %(default)s)')
    parser.add_argument('--subtitle', default='Work in progress', help='Subtitle (default: %(default)s)')
    parser.add_argument('--titlePos', type=int, default=0, help='Title position 0-4 (default: %(default)s)')
    args = parser.parse_args()

    # output path
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.fit))[0]
        args.out = f'residuals_{stem}.pdf'

    # ---- load ----
    print(f"Loading fit:  {args.fit}")
    coeffs, params_or_modes, meta = load_fit(args.fit)
    components = list(meta.get('components', ['Bz', 'Br', 'Bphi']))
    basis = meta['basis']
    if basis == 'bessel':
        lmax = f"bessel n_max={meta['n_max']} m_max={meta['m_max']} R={meta['R']:.0f}cm"
    elif basis == 'cylindrical':
        lmax = f"cyl n_max={meta['n_max']} m_max={meta['m_max']}"
    elif basis == 'zernike':
        lmax = f"n_max={meta.get('n_max','?')} l_max={meta.get('l_max','?')}"
    else:
        nms  = meta.get('n_max_sum')
        lp   = meta.get('l_max_phi')
        mmpl = meta.get('m_max_per_l_str')
        if mmpl is not None:
            lmax = f"custom ({len(meta.get('m_max_per_l') or {})} overrides)"
        elif nms is not None:
            lmax = f"diamond n_max_sum={nms}"
        elif lp is not None and lp != meta.get('l_max'):
            lmax = f"l_max={meta.get('l_max','?')} l_max_phi={lp}"
        else:
            lmax = meta.get('l_max', '?')

    # Use cuts from the npz if available; command-line --rmax-sphere overrides rmax_sphere only
    rmax_sphere = args.rmax_sphere
    if np.isnan(rmax_sphere) and not np.isnan(meta.get('rmax_sphere_cm', np.nan)):
        rmax_sphere = meta['rmax_sphere_cm']
    rmax_sphere = None if np.isnan(rmax_sphere) else rmax_sphere
    rmax_cm = None if np.isnan(meta.get('rmax_cm', np.nan)) else meta['rmax_cm']
    zmax_cm = None if np.isnan(meta.get('zmax_cm', np.nan)) else meta['zmax_cm']
    rmin_sphere = None if np.isnan(meta.get('rmin_sphere_cm', np.nan)) else meta['rmin_sphere_cm']

    excl = meta.get('exclude_vol_boundaries', False)
    print(f"Loading grid: {args.grid}  "
          f"(rmax={rmax_cm}, zmax={zmax_cm}, rmax_sphere={rmax_sphere}"
          f"{f', rmin_sphere={rmin_sphere}' if rmin_sphere is not None else ''} cm"
          f"{', excl vol boundaries' if excl else ''})")
    grid = load_grid(args.grid, rmax_cm=rmax_cm, zmax_cm=zmax_cm,
                     rmax_sphere_cm=rmax_sphere, rmin_sphere_cm=rmin_sphere,
                     exclude_vol_boundaries=excl)
    print(f"  {len(grid['r_cm']):,} points after cut")

    # ---- evaluate ----
    print("Evaluating fit ...")
    residuals = compute_residuals(grid, coeffs, params_or_modes, meta, components)

    r   = grid['r_cm']
    z   = grid['z_cm']
    phi = grid['phi']
    R_sph = np.sqrt(r**2 + z**2)

    # ---- print summary ----
    print()
    print(f"{'Component':<8}  {'RMS [mT]':>10}  {'Mean [mT]':>10}  {'|Max| [mT]':>12}")
    print("-"*46)
    for comp in components:
        res = residuals[comp]
        print(f"{comp:<8}  {np.sqrt(np.mean(res**2)):10.4f}  {np.mean(res):10.4f}  "
              f"{np.max(np.abs(res)):12.4f}")

    # ---- individual plots ----
    outdir = output_tools.make_plot_dir(os.path.dirname(os.path.abspath(args.out)))
    stem   = os.path.splitext(os.path.basename(args.out))[0]

    def _save(fig, suffix):
        _add_cms_label(fig, args)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plot_tools.save_pdf_and_png(outdir, f'{stem}_{suffix}')
        plt.close(fig)

    # ------------------------------------------------------------------
    # histograms — one figure per component (all points)
    # ------------------------------------------------------------------
    for comp in components:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        make_histogram_page(residuals, [comp], '', [ax])
        _save(fig, f'hist_{comp}')

    # ------------------------------------------------------------------
    # histograms — tracker and outer regions separately
    # ------------------------------------------------------------------
    tk_mask    = (r < TRACKER_OAE_RMAX) & (np.abs(z) < TRACKER_OAE_ZMAX)
    outer_mask = ~tk_mask
    for region_mask, region_tag, region_label in [
        (tk_mask,    'tracker', rf'$r<{TRACKER_OAE_RMAX:.0f}$ cm, $|z|<{TRACKER_OAE_ZMAX:.0f}$ cm'),
        (outer_mask, 'outer',   rf'outer region'),
    ]:
        if region_mask.sum() < 10:
            continue
        region_residuals = {comp: residuals[comp][region_mask] for comp in components}
        for comp in components:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            make_histogram_page(region_residuals, [comp], '', [ax],
                                region_label=region_label)
            _save(fig, f'hist_{comp}_{region_tag}')

    # ------------------------------------------------------------------
    # profiles vs r, z, phi, R — one figure per (variable, component)
    # ------------------------------------------------------------------
    for var, xlabel, suffix in [
        (r,     r'$r$ [cm]',       'vs_r'),
        (z,     r'$z$ [cm]',       'vs_z'),
        (phi,   r'$\phi$ [rad]',   'vs_phi'),
        (R_sph, r'$R_{sph}$ [cm]', 'vs_R'),
    ]:
        for comp in components:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            slice_plot(ax, var, residuals[comp], xlabel, comp, nbins=args.nbins)
            _save(fig, f'{suffix}_{comp}')

    # ------------------------------------------------------------------
    # profiles vs r, z, phi, R — tracker and outer regions separately
    # ------------------------------------------------------------------
    tk_mask   = (r < TRACKER_OAE_RMAX) & (np.abs(z) < TRACKER_OAE_ZMAX)
    outer_mask = ~tk_mask
    for var, xlabel, suffix in [
        (r,     r'$r$ [cm]',       'vs_r'),
        (z,     r'$z$ [cm]',       'vs_z'),
        (phi,   r'$\phi$ [rad]',   'vs_phi'),
        (R_sph, r'$R_{sph}$ [cm]', 'vs_R'),
    ]:
        for region_mask, region_tag in [(tk_mask, 'tracker'), (outer_mask, 'outer')]:
            if region_mask.sum() < 10:
                continue
            for comp in components:
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                slice_plot(ax, var[region_mask], residuals[comp][region_mask],
                           xlabel, comp, nbins=args.nbins)
                _save(fig, f'{suffix}_{comp}_{region_tag}')

    # ------------------------------------------------------------------
    # 2D maps in (r, z) — one figure per component
    # ------------------------------------------------------------------
    for comp in components:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
        make_2d_map(ax, z, r, residuals[comp], comp, r'$z$ [cm]', r'$r$ [cm]')
        ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
        add_tracker_boundaries(ax)
        _save(fig, f'map_rz_{comp}')

    # ------------------------------------------------------------------
    # 2D maps in (phi, z) — one figure per component
    # ------------------------------------------------------------------
    for comp in components:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
        make_2d_map(ax, phi, z, residuals[comp], comp, r'$\phi$ [rad]', r'$z$ [cm]')
        ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
        _save(fig, f'map_phiz_{comp}')

    # ------------------------------------------------------------------
    # 2D maps in (r, z) — tracker region only, one figure per component
    # ------------------------------------------------------------------
    tk = (r < TRACKER_OAE_RMAX) & (np.abs(z) < TRACKER_OAE_ZMAX)
    if tk.sum() > 20:
        for comp in components:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
            make_2d_map(ax, z[tk], r[tk], residuals[comp][tk], comp,
                        r'$z$ [cm]', r'$r$ [cm]', nx=56, ny=23)
            ax.set_xlim(-TRACKER_OAE_ZMAX, TRACKER_OAE_ZMAX)
            ax.set_ylim(0, TRACKER_OAE_RMAX)
            ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
            rms_tk = np.sqrt(np.mean(residuals[comp][tk]**2))
            ax.text(0.02, 0.97, f'RMS={rms_tk:.4f} mT  N={tk.sum():,}',
                    transform=ax.transAxes, fontsize='small', va='top')
            _save(fig, f'map_rz_tracker_{comp}')

    # ------------------------------------------------------------------
    # 2D maps in (phi, z) — tracker region only, one figure per component
    # ------------------------------------------------------------------
    if tk.sum() > 20:
        for comp in components:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
            make_2d_map(ax, phi[tk], z[tk], residuals[comp][tk], comp,
                        r'$\phi$ [rad]', r'$z$ [cm]', nx=36, ny=56)
            ax.set_ylim(-TRACKER_OAE_ZMAX, TRACKER_OAE_ZMAX)
            ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
            rms_tk = np.sqrt(np.mean(residuals[comp][tk]**2))
            ax.text(0.02, 0.97, f'RMS={rms_tk:.4f} mT  N={tk.sum():,}',
                    transform=ax.transAxes, fontsize='small', va='top')
            _save(fig, f'map_phiz_tracker_{comp}')

    # ------------------------------------------------------------------
    # 2D maps in (r, phi) — midplane |z| < 20 cm, one figure per component
    # ------------------------------------------------------------------
    mid = np.abs(z) < 20
    if mid.sum() > 10:
        for comp in components:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
            make_2d_map(ax, phi[mid], r[mid], residuals[comp][mid],
                        comp, r'$\phi$ [rad]', r'$r$ [cm]', nx=36, ny=20)
            ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
            _save(fig, f'map_rphi_{comp}')

    # ------------------------------------------------------------------
    # shell histograms — one figure per component
    # ------------------------------------------------------------------
    r_shells = [
        (0,   115, 280.0, 'tracker  r<115, |z|<280 cm'),
        (115, 200, None,  '115<r<200 cm'),
        (200, 280, None,  '200<r<280 cm'),
        (280, 320, None,  '280<r<320 cm  (near coil)'),
    ]

    def _shell_mask(rlo, rhi, zmax):
        m = (r >= rlo) & (r < rhi)
        if zmax is not None:
            m = m & (np.abs(z) < zmax)
        return m

    for comp in components:
        for rlo, rhi, zmax, label in r_shells:
            mask = _shell_mask(rlo, rhi, zmax)
            if mask.sum() < 10:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
            res = residuals[comp][mask]
            rms = np.sqrt(np.mean(res**2))
            bins = np.linspace(-3*rms, 3*rms, 60)
            ax.hist(res, bins=bins, color=COMP_COLORS[comp], alpha=0.75, density=True)
            ax.axvline(0, color='k', lw=0.8, ls='--')
            ax.axvline(np.mean(res), color='red', lw=1.0)
            ax.text(0.5, 0.97, label, ha='center', va='top',
                    transform=ax.transAxes, fontsize='small')
            ax.text(0.02, 0.80, f'RMS={rms:.4f} mT\nN={mask.sum():,}',
                    transform=ax.transAxes, va='top', fontsize='small')
            ax.set_xlabel('residual [mT]')
            ax.set_ylabel('density')
            ax.text(1.0, 1.0, COMP_LABELS[comp], ha='right', va='bottom', transform=ax.transAxes)
            shell_tag = f'{rlo}_{rhi}'
            _save(fig, f'shell_{comp}_{shell_tag}')

    output_tools.write_index_and_log(outdir, stem, args=args)
    print(f"Done → {outdir}")


if __name__ == '__main__':
    main()
