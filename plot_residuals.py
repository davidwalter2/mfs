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

# Tracker volume boundaries (from CMSSW ParametrizedEngine isDefined())
# OAE (default parametrization, standard tracking volume): r < 115 cm, |z| < 280 cm
# PolyFit3D (CVH refit validity, extended): r < 190 cm, |z| < 350 cm, |z|+2.5r < 670 cm
TRACKER_OAE_RMAX  = 115.0   # cm
TRACKER_OAE_ZMAX  = 280.0   # cm
TRACKER_P3D_RMAX  = 190.0   # cm
TRACKER_P3D_ZMAX  = 350.0   # cm  (also corner cut |z|+2.5r < 670 cm)


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
        linestyle='-', label=f'OAE tracker (r<{TRACKER_OAE_RMAX:.0f}, |z|<{TRACKER_OAE_ZMAX:.0f} cm)',
        zorder=5,
    )
    # PolyFit3D / CVH validity rectangle
    rect_p3d = mpatches.Rectangle(
        (-TRACKER_P3D_ZMAX, 0),
        2 * TRACKER_P3D_ZMAX, TRACKER_P3D_RMAX,
        linewidth=1.5, edgecolor='green', facecolor='none',
        linestyle='--', label=f'PolyFit3D (r<{TRACKER_P3D_RMAX:.0f}, |z|<{TRACKER_P3D_ZMAX:.0f} cm)',
        zorder=5,
    )
    ax.add_patch(rect_oae)
    ax.add_patch(rect_p3d)
    ax.legend(fontsize=7, loc='upper right')

from harmonic_basis import eval_field
from cylindrical_basis import eval_field_cyl, parse_mode_label
from fit_field import load_grid


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
    basis = str(d['basis']) if 'basis' in d else 'spherical'

    if basis == 'cylindrical':
        modes = [parse_mode_label(s) for s in param_strs]
        meta = {
            'basis': 'cylindrical',
            'L':     float(d['L']),
            'n_max': int(d['n_max']),
            'm_max': int(d['m_max']),
            'components': list(d.get('components', ['Bz', 'Br', 'Bphi'])),
        }
        return coeffs, modes, meta
    else:
        params = []
        for s in param_strs:
            parts = s.split('_')
            params.append((int(parts[0][1:]), int(parts[1][1:]), parts[2]))
        def _scalar(key, default=np.nan):
            return float(d[key]) if key in d else default

        meta = {
            'basis':          'spherical',
            'r_scale':        float(d['r_scale']),
            'z0':             float(d['z0']) if 'z0' in d else 0.0,
            'l_max':          int(d['l_max']) if 'l_max' in d else -1,
            'components':     list(d.get('components', ['Bz', 'Br', 'Bphi'])),
            'rmax_cm':        _scalar('rmax_cm'),
            'zmax_cm':        _scalar('zmax_cm'),
            'rmax_sphere_cm': _scalar('rmax_sphere_cm'),
        }
        return coeffs, params, meta


def compute_residuals(grid, coeffs, params_or_modes, meta,
                      components=('Bz', 'Br', 'Bphi')):
    """Return dict of residual arrays (measured - predicted) in mT."""
    if meta['basis'] == 'cylindrical':
        pred = eval_field_cyl(coeffs, params_or_modes,
                              grid['r_cm'], grid['phi'], grid['z_cm'],
                              L=meta['L'], components=components)
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


def make_histogram_page(residuals, components, title, ax_arr):
    """Fill a row of histogram axes, one per component."""
    for ax, comp in zip(ax_arr, components):
        res = residuals[comp]
        rms = np.sqrt(np.mean(res**2))
        mu  = np.mean(res)
        bins = np.linspace(-3*rms, 3*rms, 80)
        ax.hist(res, bins=bins, color=COMP_COLORS[comp], alpha=0.75, density=True)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axvline(mu,  color='red',  lw=1.0, ls='-', label=f'mean={mu:+.3f} mT')
        ax.set_xlabel(f'{COMP_LABELS[comp]} residual [mT]')
        ax.set_ylabel('density')
        ax.set_title(f'{COMP_LABELS[comp]}  RMS={rms:.4f} mT  N={len(res):,}')
        ax.legend(fontsize=8)


def slice_plot(fig, gs_row, x_vals, res_vals, xlabel, comp, nbins=30,
               profile_only=False):
    """
    One column of a slice row: scatter (or profile mean ± RMS) vs x_vals.
    gs_row: a 2-element GridSpec row [scatter_ax, profile_ax].
    """
    ax = fig.add_subplot(gs_row)
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
    ax.set_ylabel(f'{COMP_LABELS[comp]} residual [mT]')
    ax.legend(fontsize=7)
    return ax


def make_slice_page(fig, residuals, grid, components, var, xlabel, nbins=30):
    """
    One page of slice plots: rows=components, 1 profile plot per component.
    Returns the axes list.
    """
    n = len(components)
    gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.45)
    for i, comp in enumerate(components):
        slice_plot(fig, gs[i], var, residuals[comp], xlabel, comp, nbins=nbins)


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
    parser.add_argument('--fit', default='tosca170812_extended_coeffs_lmax18_all3_rmax290.npz',
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
    if basis == 'cylindrical':
        lmax = f"cyl n_max={meta['n_max']} m_max={meta['m_max']}"
    else:
        lmax = meta.get('l_max', '?')

    # Use cuts from the npz if available; command-line --rmax-sphere overrides rmax_sphere only
    rmax_sphere = args.rmax_sphere
    if np.isnan(rmax_sphere) and not np.isnan(meta.get('rmax_sphere_cm', np.nan)):
        rmax_sphere = meta['rmax_sphere_cm']
    rmax_cm = None if np.isnan(meta.get('rmax_cm', np.nan)) else meta['rmax_cm']
    zmax_cm = None if np.isnan(meta.get('zmax_cm', np.nan)) else meta['zmax_cm']

    print(f"Loading grid: {args.grid}  "
          f"(rmax={rmax_cm}, zmax={zmax_cm}, rmax_sphere={rmax_sphere} cm)")
    grid = load_grid(args.grid, rmax_cm=rmax_cm, zmax_cm=zmax_cm,
                     rmax_sphere_cm=rmax_sphere)
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

    # ---- multi-page PDF ----
    from matplotlib.backends.backend_pdf import PdfPages
    print(f"\nWriting {args.out} ...")
    with PdfPages(args.out) as pdf:

        # ------------------------------------------------------------------
        # Page 1: overall histograms
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, len(components), figsize=(5*len(components), 4.5))
        if len(components) == 1:
            axes = [axes]
        fig.suptitle(f'Harmonic fit residuals (l_max={lmax}, rmax_sphere={args.rmax_sphere} cm)\n'
                     f'{os.path.basename(args.fit)}', fontsize=10)
        make_histogram_page(residuals, components, '', axes)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2: profiles vs r
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(5*len(components), 4.5))
        fig.suptitle(f'Residuals vs r  (l_max={lmax})', fontsize=10)
        make_slice_page(fig, residuals, grid, components,
                        r, r'$r$ [cm]', nbins=args.nbins)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 3: profiles vs z
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(5*len(components), 4.5))
        fig.suptitle(f'Residuals vs z  (l_max={lmax})', fontsize=10)
        make_slice_page(fig, residuals, grid, components,
                        z, r'$z$ [cm]', nbins=args.nbins)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 4: profiles vs phi
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(5*len(components), 4.5))
        fig.suptitle(f'Residuals vs $\\phi$  (l_max={lmax})', fontsize=10)
        make_slice_page(fig, residuals, grid, components,
                        phi, r'$\phi$ [rad]', nbins=args.nbins)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 5: profiles vs spherical R
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(5*len(components), 4.5))
        fig.suptitle(f'Residuals vs spherical $R$  (l_max={lmax})', fontsize=10)
        make_slice_page(fig, residuals, grid, components,
                        R_sph, r'$R_{sph}$ [cm]', nbins=args.nbins)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 6: 2D maps in (r, z) for each component
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, len(components), figsize=(6*len(components), 5))
        if len(components) == 1:
            axes = [axes]
        fig.suptitle(f'Mean residual map in $(r, z)$  (l_max={lmax})', fontsize=10)
        for ax, comp in zip(axes, components):
            make_2d_map(ax, z, r, residuals[comp], comp, r'$z$ [cm]', r'$r$ [cm]')
            ax.set_title(COMP_LABELS[comp])
            add_tracker_boundaries(ax)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 7: 2D maps in (phi, z)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, len(components), figsize=(6*len(components), 5))
        if len(components) == 1:
            axes = [axes]
        fig.suptitle(f'Mean residual map in $(\phi, z)$  (l_max={lmax})', fontsize=10)
        for ax, comp in zip(axes, components):
            make_2d_map(ax, phi, z, residuals[comp], comp,
                        r'$\phi$ [rad]', r'$z$ [cm]')
            ax.set_title(COMP_LABELS[comp])
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 8: 2D maps in (r, phi) — midplane only (|z| < 20 cm)
        # ------------------------------------------------------------------
        mid = np.abs(z) < 20
        if mid.sum() > 10:
            fig, axes = plt.subplots(1, len(components), figsize=(6*len(components), 5))
            if len(components) == 1:
                axes = [axes]
            fig.suptitle(f'Mean residual map in $(r, \\phi)$, $|z|<20$ cm  (l_max={lmax})',
                         fontsize=10)
            for ax, comp in zip(axes, components):
                make_2d_map(ax, phi[mid], r[mid], {comp: residuals[comp][mid]}[comp],
                            comp, r'$\phi$ [rad]', r'$r$ [cm]', nx=36, ny=20)
                ax.set_title(COMP_LABELS[comp])
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------------
        # Page 9: residuals in r shells (tracker, mid, outer, coil)
        # ------------------------------------------------------------------
        # Shell definitions: (r_lo, r_hi, z_max_or_None, label)
        # The tracker shell applies both r<115 AND |z|<280 (OAE validity region).
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
            n_shell = sum(1 for rlo, rhi, zmax, _ in r_shells
                          if _shell_mask(rlo, rhi, zmax).sum() > 10)
            fig, axes = plt.subplots(1, n_shell, figsize=(5*n_shell, 4))
            if n_shell == 1:
                axes = [axes]
            fig.suptitle(f'{COMP_LABELS[comp]} residuals in r shells  (l_max={lmax})',
                         fontsize=10)
            ax_idx = 0
            for rlo, rhi, zmax, label in r_shells:
                mask = _shell_mask(rlo, rhi, zmax)
                if mask.sum() < 10:
                    continue
                ax = axes[ax_idx]; ax_idx += 1
                res = residuals[comp][mask]
                rms = np.sqrt(np.mean(res**2))
                bins = np.linspace(-3*rms, 3*rms, 60)
                ax.hist(res, bins=bins, color=COMP_COLORS[comp], alpha=0.75, density=True)
                ax.axvline(0, color='k', lw=0.8, ls='--')
                ax.axvline(np.mean(res), color='red', lw=1.0)
                ax.set_title(f'{label}\nRMS={rms:.4f} mT  N={mask.sum():,}')
                ax.set_xlabel(f'{COMP_LABELS[comp]} residual [mT]')
                ax.set_ylabel('density')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig); plt.close(fig)

    print(f"Done → {args.out}")


if __name__ == '__main__':
    main()
