"""
plot_corr.py  —  Plot the parameter correlation matrix for a spherical-harmonic fit.

The correlation matrix is derived from the Gram matrix of the design matrix:
  C_ij = [A^T A]_ij / sqrt([A^T A]_ii * [A^T A]_jj)

This equals the parameter correlation matrix (up to the unknown noise variance σ²,
which cancels in the normalisation).

Axis ordering matches the param list: l=1..l_max, for each l m=0..l (cosine then sine).
Block boundaries are drawn at each new l value.

Usage
-----
  python plot_corr.py --fit <file.npz> --grid <field_grid.txt> [--out corr.pdf]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fit_field import load_grid, CMSSW_VOL_BOUNDARIES_CM, parse_m_max_per_l
from harmonic_basis import build_design_matrix, param_list
from cylindrical_basis import build_design_matrix_cyl, parse_mode_label
from cylindrical_bessel_basis import build_design_matrix_cjb
import zernike_basis
import os
import mplhep as hep
from wums import output_tools, plot_tools
hep.style.use(hep.style.ROOT)


def build_block_boundaries(l_max):
    """Return list of (index, l_label) for the start of each l-block."""
    idx = 0
    boundaries = []
    for l in range(1, l_max + 1):
        boundaries.append((idx, l))
        idx += 2 * l + 1   # m=0 (1 param) + m=1..l (2 params each)
    return boundaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit',  required=True, help='Input .npz fit file')
    ap.add_argument('--grid', required=True, help='Field grid .txt file')
    ap.add_argument('--title', default='CMS', help='Experiment label (default: %(default)s)')
    ap.add_argument('--subtitle', default='Simulation', help='Subtitle (default: %(default)s)')
    ap.add_argument('--titlePos', type=int, default=0, help='Title position 0-4 (default: %(default)s)')
    ap.add_argument('-o', '--outpath', default='.', help='Output directory')
    args = ap.parse_args()

    # ── load fit metadata ────────────────────────────────────────────────────
    data       = np.load(args.fit, allow_pickle=True)
    basis_type = str(data.get('basis_type', 'harmonic'))
    components = list(data['components'])

    def _float(key, default=None):
        if key not in data: return default
        v = float(data[key])
        return None if np.isnan(v) else v

    rmax_cm     = _float('rmax_cm')
    zmax_cm     = _float('zmax_cm')
    rsph_cm     = _float('rmax_sphere_cm')
    rmin_sph_cm = _float('rmin_sphere_cm')
    excl        = bool(data['exclude_vol_boundaries']) if 'exclude_vol_boundaries' in data else False
    print(f'cuts: r<{rmax_cm}, |z|<{zmax_cm}, R_sph<{rsph_cm}, excl={excl}')

    # ── load grid ────────────────────────────────────────────────────────────
    grid = load_grid(args.grid,
                     rmax_cm=rmax_cm, zmax_cm=zmax_cm, rmax_sphere_cm=rsph_cm,
                     rmin_sphere_cm=rmin_sph_cm, exclude_vol_boundaries=excl)
    r = grid['r_cm']; z = grid['z_cm']; phi = grid['phi']
    n = len(r)
    print(f'{n} grid points loaded')

    # ── build design matrix ──────────────────────────────────────────────────
    print('Building design matrix ...')
    if basis_type == 'bessel':
        n_max = int(data['n_max'])
        m_max = int(data['m_max'])
        R     = float(data['R'])
        print(f'J-Bessel: n_max={n_max}, m_max={m_max}, R={R:.1f} cm')
        A, modes_rebuilt = build_design_matrix_cjb(
            r, phi, z, n_max, m_max, R, components=tuple(components))
    elif basis_type == 'cylindrical':
        n_max  = int(data['n_max'])
        m_max  = int(data['m_max'])
        L      = float(data['L'])
        print(f'Cylindrical: n_max={n_max}, m_max={m_max}, L={L:.1f} cm')
        A, modes_rebuilt = build_design_matrix_cyl(
            r, phi, z, n_max, m_max, L, components=tuple(components))
    elif basis_type == 'zernike':
        r_scale = float(data['r_scale'])
        z0      = float(data['z0']) if 'z0' in data else 0.0
        n_max   = int(data['n_max']); l_max = int(data['l_max'])
        print(f'Zernike: n_max={n_max}, l_max={l_max}, r_scale={r_scale:.1f} cm')
        A, modes_rebuilt, _ = zernike_basis.build_design_matrix(
            r, phi, z, n_max, l_max,
            components=tuple(components), r_scale=r_scale, z0=z0)
    else:
        r_scale     = float(data['r_scale'])
        z0          = float(data['z0']) if 'z0' in data else 0.0
        l_max       = int(data['l_max'])
        l_max_phi   = int(data['l_max_phi'])   if 'l_max_phi'   in data else None
        n_max_sum   = int(data['n_max_sum'])   if 'n_max_sum'   in data else None
        mmpl_str    = str(data['m_max_per_l_str']) if 'm_max_per_l_str' in data else None
        m_max_per_l = parse_m_max_per_l(mmpl_str) if mmpl_str else None
        lp_str = f', l_max_phi={l_max_phi}' if l_max_phi is not None else ''
        print(f'Harmonic: l_max={l_max}{lp_str}, r_scale={r_scale:.1f} cm')
        A, modes_rebuilt, _ = build_design_matrix(
            r, phi, z, l_max,
            components=tuple(components), r_scale=r_scale, z0=z0,
            l_max_phi=l_max_phi, n_max_sum=n_max_sum, m_max_per_l=m_max_per_l)
    print(f'Design matrix shape: {A.shape}')

    # ── Gram matrix → correlation ─────────────────────────────────────────────
    print('Computing Gram matrix ...')
    G    = A.T @ A
    diag = np.sqrt(np.diag(G))
    diag = np.where(diag > 0, diag, 1.0)
    C    = G / np.outer(diag, diag)
    n_params = C.shape[0]
    print(f'Correlation matrix: {n_params}x{n_params}')

    # ── reorder modes so same-group modes are contiguous ─────────────────────
    # The design matrix column order may interleave modes from different l/n
    # groups. Sort by (group_key, subkey) so the correlation matrix has clean
    # diagonal blocks when visualised.
    if basis_type in ('cylindrical', 'bessel'):
        sort_key = lambda i: (modes_rebuilt[i][0], modes_rebuilt[i][1], modes_rebuilt[i][2], modes_rebuilt[i][3])
    elif basis_type == 'zernike':
        sort_key = lambda i: (modes_rebuilt[i][0], modes_rebuilt[i][1], modes_rebuilt[i][2], modes_rebuilt[i][3])
    else:  # harmonic: (l, m, cs)
        cs_order = {'c': 0, 's': 1}
        sort_key = lambda i: (modes_rebuilt[i][0], modes_rebuilt[i][1], cs_order.get(modes_rebuilt[i][2], 0))
    perm = sorted(range(n_params), key=sort_key)
    C = C[np.ix_(perm, perm)]
    modes_rebuilt = [modes_rebuilt[i] for i in perm]

    # ── block boundaries for axis labelling ──────────────────────────────────
    if basis_type in ('cylindrical', 'bessel'):
        # Group by n (wavenumber index)
        block_index = {}
        for idx, (nv, mv, cphi, cz) in enumerate(modes_rebuilt):
            if nv not in block_index:
                block_index[nv] = idx
        blocks_sorted  = sorted(block_index.items())
        block_starts   = [b[1] for b in blocks_sorted]
        block_labels   = [b[0] for b in blocks_sorted]
        tick_positions = []
        for i, (_, start) in enumerate(blocks_sorted):
            end = blocks_sorted[i+1][1] if i+1 < len(blocks_sorted) else n_params
            tick_positions.append((start + end) / 2.0)
        xlabel = r'mode index  (grouped by $n$)'
        ylabel = r'mode index  (grouped by $n$)'
        title_extra = f'$n_{{\\max}}={n_max}$, $m_{{\\max}}={m_max}$'
        tick_fmt = lambda v: f'$n={v}$'
    elif basis_type == 'zernike':
        block_index = {}
        for idx, (nv, lv, mv, csv) in enumerate(modes_rebuilt):
            if nv not in block_index:
                block_index[nv] = idx
        blocks_sorted  = sorted(block_index.items())
        block_starts   = [b[1] for b in blocks_sorted]
        block_labels   = [b[0] for b in blocks_sorted]
        tick_positions = []
        for i, (_, start) in enumerate(blocks_sorted):
            end = blocks_sorted[i+1][1] if i+1 < len(blocks_sorted) else n_params
            tick_positions.append((start + end) / 2.0)
        xlabel = r'mode index  (grouped by $n$)'
        ylabel = r'mode index  (grouped by $n$)'
        title_extra = f'$n_{{\\max}}={n_max}$, $\\ell_{{\\max}}={l_max}$'
        tick_fmt = lambda v: f'$n={v}$'
    else:
        # Derive block boundaries from the actual params list (handles custom truncations)
        block_index = {}
        for idx, (lv, mv, csv) in enumerate(modes_rebuilt):
            if lv not in block_index:
                block_index[lv] = idx
        blocks_sorted  = sorted(block_index.items())   # [(l, start_idx), ...]
        block_starts   = [b[1] for b in blocks_sorted]
        block_labels   = [b[0] for b in blocks_sorted]
        tick_positions = []
        for i, (lv, start) in enumerate(blocks_sorted):
            end = blocks_sorted[i+1][1] if i+1 < len(blocks_sorted) else n_params
            tick_positions.append((start + end) / 2.0)
        xlabel = r'mode index  (grouped by $\ell$)'
        ylabel = r'mode index  (grouped by $\ell$)'
        title_extra = f'$\\ell_{{\\max}}={l_max}$'
        tick_fmt = lambda v: f'$\\ell={v}$'

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(C, origin='upper', aspect='equal',
                   cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')

    # Fix MAXTICKS: ROOT style enables AutoMinorLocator which generates O(1000) ticks on
    # a 360-element image axis; replace with FixedLocator + NullLocator immediately after imshow
    ax.xaxis.set_major_locator(plot_tools.ticker.FixedLocator(tick_positions))
    ax.yaxis.set_major_locator(plot_tools.ticker.FixedLocator(tick_positions))
    ax.xaxis.set_minor_locator(plot_tools.ticker.NullLocator())
    ax.yaxis.set_minor_locator(plot_tools.ticker.NullLocator())

    for start in block_starts[1:]:
        ax.axvline(start - 0.5, color='black', lw=0.4, alpha=0.5)
        ax.axhline(start - 0.5, color='black', lw=0.4, alpha=0.5)

    ax.set_xticklabels([tick_fmt(v) for v in block_labels], rotation=90)
    ax.set_yticklabels([tick_fmt(v) for v in block_labels])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('correlation')

    fig.tight_layout()
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos, no_energy=True)
    outdir = output_tools.make_plot_dir(args.outpath)
    stem = os.path.splitext(os.path.basename(args.fit))[0]
    outfile = f'corr_{stem}'
    plot_tools.save_pdf_and_png(outdir, outfile)
    output_tools.write_index_and_log(outdir, outfile, args=args)


if __name__ == '__main__':
    main()
