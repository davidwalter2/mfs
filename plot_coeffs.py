"""
plot_coeffs.py  —  Visualise fit coefficients stored in a .npz file.

Supports both the spherical-harmonic basis and the Zernike basis.

Harmonic layout
---------------
  x-axis : m = -l_max … +l_max
            m < 0  →  sine   (B_{l,|m|}) coefficients
            m = 0  →  axisymmetric (A_{l,0})
            m > 0  →  cosine (A_{l,m}) coefficients
  y-axis : l = 1 … l_max  (l=1 at the top)

Zernike layout
--------------
  x-axis : signed m  (same convention as harmonic)
  y-axis : one row per unique (n, l) pair, ordered by (l, n)
            — n=l rows (standard harmonic modes) are highlighted

Each valid cell is drawn as a filled circle whose
  • size   ∝  log10(|coeff|)  (clamped to [1, 400] pt²)
  • colour encodes the coefficient value on a symmetric log scale

Text annotations are placed for |coeff| > TEXT_THRESHOLD.

Usage
-----
  python plot_coeffs.py --fit <file.npz> [--out coeffs.pdf] [--text-threshold 1e-3]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ── text threshold: only annotate cells whose |coeff| exceeds this ─────────
DEFAULT_TEXT_THRESHOLD = 1e-4

# ── bubble size mapping ─────────────────────────────────────────────────────
SIZE_LOG_MIN = -6     # log10(|coeff|) mapped to minimum dot size
SIZE_LOG_MAX = 3      # log10(|coeff|) mapped to maximum dot size
SIZE_PTS_MIN = 8      # pt²  (smallest visible dot)
SIZE_PTS_MAX = 1000   # pt²  (largest dot)


def log_size(val, log_min=SIZE_LOG_MIN, log_max=SIZE_LOG_MAX,
             pts_min=SIZE_PTS_MIN, pts_max=SIZE_PTS_MAX):
    """Map |val| to a marker size in pt², using a log scale."""
    with np.errstate(divide='ignore', invalid='ignore'):
        lv = np.where(np.abs(val) > 0, np.log10(np.abs(val)), log_min - 1)
    t = np.clip((lv - log_min) / (log_max - log_min), 0, 1)
    return pts_min + t * (pts_max - pts_min)


def fmt(v, threshold=1e-2):
    """Format a coefficient value compactly."""
    a = abs(v)
    if a == 0:
        return '0'
    if a >= threshold:
        return f'{v:.2f}'
    # scientific notation, 2 significant figures
    exp = int(np.floor(np.log10(a)))
    mant = v / 10**exp
    return f'{mant:.1f}e{exp}'


def parse_params_cyl(params_arr):
    """Convert array of strings 'n{n}_m{m}_{cphi}_{cz}' to list of (n, m, cphi, cz)."""
    out = []
    for s in params_arr:
        parts = str(s).split('_')
        n    = int(parts[0][1:])
        m    = int(parts[1][1:])
        cphi = parts[2]
        cz   = parts[3]
        out.append((n, m, cphi, cz))
    return out


def parse_params_harmonic(params_arr):
    """Convert array of strings 'l{l}_m{m}_{cs}' to list of (l, m, cs)."""
    out = []
    for s in params_arr:
        parts = str(s).split('_')
        l = int(parts[0][1:])
        m = int(parts[1][1:])
        cs = parts[2]
        out.append((l, m, cs))
    return out


def parse_params_zernike(params_arr):
    """Convert array of strings 'n{n}_l{l}_m{m}_{cs}' to list of (n, l, m, cs)."""
    out = []
    for s in params_arr:
        parts = str(s).split('_')
        n  = int(parts[0][1:])
        l  = int(parts[1][1:])
        m  = int(parts[2][1:])
        cs = parts[3]
        out.append((n, l, m, cs))
    return out


def _inset_colorbar(fig, ax, sc, label, fontsize=14):
    """Place a colorbar inside the axes frame (upper-right corner)."""
    cax = ax.inset_axes([0.91, 0.46, 0.025, 0.50])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    return cbar


def _scatter_and_annotate(ax, xs, ys, vals, norm, cmap, text_thresh, fontsize=8):
    """Shared helper: black anchor dots, coloured bubbles, and text annotations.

    A small black dot is drawn at every position (zorder=1) to mark modes that
    are part of the basis (allowed to float), even when their fitted value is
    near zero.  The coloured bubble sits on top (zorder=3).
    """
    ax.scatter(xs, ys, s=30, c='black', zorder=1, linewidths=0)
    sizes = log_size(vals)
    sc = ax.scatter(xs, ys, s=sizes, c=vals, cmap=cmap, norm=norm,
                    edgecolors='none', zorder=3)
    for x, y, v in zip(xs, ys, vals):
        if abs(v) > text_thresh:
            fc = cmap(norm(v))
            lum = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc = 'white' if lum < 0.5 else 'black'
            ax.text(x, y, fmt(v), ha='center', va='center',
                    fontsize=fontsize, color=tc, zorder=4, fontweight='normal')
    return sc


def plot_cylindrical(coeffs, modes, n_max, m_max, fit_path, text_thresh):
    """Bubble plot for the cylindrical Fourier-Bessel basis.

    x-axis: signed n  — positive = z-symmetric (cz='sym'), negative = z-antisymmetric (cz='anti')
    y-axis: signed m  — positive = cosine (cphi='c'), negative = sine (cphi='s')

    Convention mirrors the harmonic plot: right/up = 'cosine-like', left/down = 'sine-like'.
    The n=0 uniform mode sits at (x=0, y=0).
    """
    coeff_of = {(n, m, cphi, cz): c for c, (n, m, cphi, cz) in zip(coeffs, modes)}

    xs, ys, vals = [], [], []
    for (n, m, cphi, cz) in modes:
        x = n  if cz   == 'sym' else -n
        y = m  if cphi == 'c'   else -m
        xs.append(x); ys.append(y); vals.append(coeff_of[(n, m, cphi, cz)])
    xs   = np.array(xs,   dtype=float)
    ys   = np.array(ys,   dtype=float)
    vals = np.array(vals, dtype=float)

    vmax = np.max(np.abs(vals)) if len(vals) else 1.0
    linthresh = max(vmax * 1e-6, 1e-10)
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    cmap = plt.get_cmap('RdBu_r')

    fig, ax = plt.subplots(figsize=(16, 9))

    sc = _scatter_and_annotate(ax, xs, ys, vals, norm, cmap, text_thresh)

    ax.set_xlim(-n_max - 0.7, n_max + 0.7)
    ax.set_ylim(-m_max - 0.7, m_max + 0.7)
    xticks = list(range(-n_max, n_max + 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(abs(x)) if x % 2 == 0 else '' for x in xticks], fontsize=14)
    yticks = list(range(-m_max, m_max + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(abs(y)) for y in yticks], fontsize=14)

    ax.set_xlabel(r'$n$  (negative = z-antisymmetric,  positive = z-symmetric)', fontsize=16)
    ax.set_ylabel(r'$m$  (negative = sine,  positive = cosine)', fontsize=16)
    ax.set_title(
        f'Cylindrical coefficients   $n_{{\\max}}={n_max}$, $m_{{\\max}}={m_max}$   '
        f'({len(coeffs)} modes)\n'
        f'{fit_path}\n'
        r'Bubble $\propto \log_{10}|c|$,  colour = signed value  (symlog)',
        fontsize=13)

    for xv in [-0.5, 0.5]:
        ax.axvline(xv, color='grey', lw=0.6, ls='--', alpha=0.6)
    for yv in [-0.5, 0.5]:
        ax.axhline(yv, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.text(-n_max * 0.5, -0.04, r'anti ($z$-odd)',
            transform=ax.get_xaxis_transform(), ha='center', fontsize=14, color='grey')
    ax.text( n_max * 0.5, -0.04, r'sym ($z$-even)',
            transform=ax.get_xaxis_transform(), ha='center', fontsize=14, color='grey')

    ax.set_axisbelow(True)
    ax.grid(True, which='both', lw=0.3, alpha=0.4)
    _inset_colorbar(fig, ax, sc, 'coefficient value')
    fig.tight_layout()
    return fig


def plot_harmonic(coeffs, params, l_max, fit_path, text_thresh):
    """Bubble plot for the spherical-harmonic basis.

    Only modes present in ``params`` are plotted — modes outside the truncation
    (e.g. m>0 for l>l_max_phi in an asymmetric fit) are simply absent rather
    than shown as zero-valued bubbles.  A black dot marks every basis position.
    """
    A = np.zeros((l_max + 1, l_max + 1))
    B = np.zeros((l_max + 1, l_max + 1))
    for c, (l, m, cs) in zip(coeffs, params):
        if cs == 'c':
            A[l, m] = c
        else:
            B[l, m] = c

    # Build scatter data only for modes that are in the basis (params), not the
    # full triangular region — so excluded modes don't appear as zero bubbles.
    xs, ys, vals = [], [], []
    for (l, m, cs) in params:
        signed_m = m if cs == 'c' else -m
        xs.append(signed_m)
        ys.append(l)
        vals.append(A[l, m] if cs == 'c' else B[l, m])
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    vals = np.array(vals, dtype=float)

    vmax = np.max(np.abs(vals))
    linthresh = max(vmax * 1e-6, 1e-10)
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    cmap = plt.get_cmap('RdBu_r')

    m_max = max(m for (l, m, cs) in params) if params else l_max
    x_range = m_max  # x-axis spans only the m values actually present

    fig, ax = plt.subplots(figsize=(16, 9))

    sc = _scatter_and_annotate(ax, xs, ys, vals, norm, cmap, text_thresh)

    ax.set_xlim(-x_range - 0.7, x_range + 0.7)
    ax.set_ylim(0.3, l_max + 0.7)
    ax.invert_yaxis()
    xticks = list(range(-x_range, x_range + 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) if abs(x) % 2 == 0 else '' for x in xticks], fontsize=15)
    ax.set_yticks(range(1, l_max + 1))
    ax.set_yticklabels([str(l) for l in range(1, l_max + 1)], fontsize=15)
    ax.set_xlabel(r'$m$  (negative = sine / $B_{l,m}$,  positive = cosine / $A_{l,m}$)',
                  fontsize=17)
    ax.set_ylabel(r'$\ell$', fontsize=18)
    ax.set_title(f'Spherical-harmonic coefficients   (file: {fit_path})\n'
                 r'Bubble size $\propto \log_{10}|c|$,  colour = signed value  (symlog scale)',
                 fontsize=14)
    for xv in [-0.5, 0.5]:
        ax.axvline(xv, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.text(-x_range * 0.5, 0.15, r'sine ($B_{l,m}$)',
            transform=ax.get_xaxis_transform(), ha='center', fontsize=15, color='grey')
    ax.text( x_range * 0.5, 0.15, r'cosine ($A_{l,m}$)',
            transform=ax.get_xaxis_transform(), ha='center', fontsize=15, color='grey')
    ax.set_axisbelow(True)
    ax.grid(True, which='both', lw=0.3, alpha=0.4)
    _inset_colorbar(fig, ax, sc, 'coefficient value')
    fig.tight_layout()
    return fig


def plot_zernike(coeffs, params, n_max, l_max, fit_path, text_thresh):
    """Bubble plot for the Zernike basis.

    y-axis: one row per unique (n, l) pair, sorted by (l, n) so that rows with
            the same angular order l are grouped together.  Rows with n=l
            (the standard harmonic modes, k=0) are highlighted with a grey band.
    x-axis: signed m  (same convention as harmonic: negative = sine, positive = cosine).
    """
    # ── collect all (n, l) pairs present in the fit, sorted by (l, n) ───────
    nl_pairs_set = set()
    for (n, l, m, cs) in params:
        nl_pairs_set.add((n, l))
    nl_pairs = sorted(nl_pairs_set, key=lambda nl: (nl[1], nl[0]))  # sort (l, n)

    row_of = {nl: i for i, nl in enumerate(nl_pairs)}  # (n,l) → row index
    n_rows = len(nl_pairs)

    # ── coefficient lookup ────────────────────────────────────────────────────
    coeff_of = {}  # (n, l, m, cs) → value
    for c, (n, l, m, cs) in zip(coeffs, params):
        coeff_of[(n, l, m, cs)] = c

    # ── build scatter data ────────────────────────────────────────────────────
    xs, ys, vals = [], [], []
    for (n, l) in nl_pairs:
        row = row_of[(n, l)]
        for m in range(0, l + 1):
            xs.append(m);  ys.append(row); vals.append(coeff_of.get((n, l, m, 'c'), 0.0))
            if m > 0:
                xs.append(-m); ys.append(row); vals.append(coeff_of.get((n, l, m, 's'), 0.0))
    xs   = np.array(xs,   dtype=float)
    ys   = np.array(ys,   dtype=float)
    vals = np.array(vals, dtype=float)

    vmax = np.max(np.abs(vals)) if len(vals) else 1.0
    linthresh = max(vmax * 1e-6, 1e-10)
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    cmap = plt.get_cmap('RdBu_r')

    fig, ax = plt.subplots(figsize=(16, 9))

    # ── highlight n=l rows (standard harmonic modes) ─────────────────────────
    for i, (n, l) in enumerate(nl_pairs):
        if n == l:
            ax.axhspan(i - 0.45, i + 0.45, color='#e8e8ff', zorder=0, lw=0)

    # ── draw l-group separators ───────────────────────────────────────────────
    prev_l = nl_pairs[0][1] if nl_pairs else None
    for i, (n, l) in enumerate(nl_pairs):
        if l != prev_l and i > 0:
            ax.axhline(i - 0.5, color='black', lw=0.8, ls='-', alpha=0.4)
        prev_l = l

    sc = _scatter_and_annotate(ax, xs, ys, vals, norm, cmap, text_thresh, fontsize=7)

    # ── y-axis labels: "n=N, ℓ=L" ────────────────────────────────────────────
    ax.set_yticks(range(n_rows))
    ylabels = []
    for (n, l) in nl_pairs:
        label = f'$n={n},\\ \\ell={l}$'
        if n == l:
            label += ' *'   # mark harmonic modes
        ylabels.append(label)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    # ── x-axis ────────────────────────────────────────────────────────────────
    xticks = list(range(-l_max, l_max + 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) if abs(x) % 2 == 0 else '' for x in xticks], fontsize=14)
    ax.set_xlim(-l_max - 0.7, l_max + 0.7)
    ax.set_xlabel(r'$m$  (negative = sine,  positive = cosine)', fontsize=16)

    for xv in [-0.5, 0.5]:
        ax.axvline(xv, color='grey', lw=0.6, ls='--', alpha=0.6)
    ax.text(-l_max * 0.5, -0.04,  r'sine',
            transform=ax.get_xaxis_transform(), ha='center', fontsize=14, color='grey')
    ax.text( l_max * 0.5, -0.04, r'cosine',
            transform=ax.get_xaxis_transform(), ha='center', fontsize=14, color='grey')

    ax.set_title(
        f'Zernike coefficients   $n_{{\\max}}={n_max}$, $\\ell_{{\\max}}={l_max}$   '
        f'({len(coeffs)} modes)\n'
        f'{fit_path}\n'
        r'Bubble $\propto \log_{10}|c|$,  colour = signed value  (symlog)'
        r'  * = standard harmonic mode ($n=\ell$)',
        fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(True, which='both', lw=0.3, alpha=0.4)

    _inset_colorbar(fig, ax, sc, 'coefficient value')
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', required=True, help='Input .npz file')
    ap.add_argument('--out', default='coeffs.pdf', help='Output PDF/PNG')
    ap.add_argument('--text-threshold', type=float, default=DEFAULT_TEXT_THRESHOLD,
                    help='Show text for |coeff| > threshold (default %(default)s)')
    args = ap.parse_args()

    data       = np.load(args.fit, allow_pickle=True)
    coeffs     = data['coeffs']
    basis_type = str(data.get('basis_type', 'harmonic'))

    if basis_type in ('cylindrical', 'bessel'):
        n_max  = int(data['n_max'])
        m_max  = int(data['m_max'])
        modes  = parse_params_cyl(data['params'])
        fig    = plot_cylindrical(coeffs, modes, n_max, m_max, args.fit, args.text_threshold)
    elif basis_type == 'zernike':
        n_max  = int(data['n_max'])
        l_max  = int(data['l_max'])
        params = parse_params_zernike(data['params'])
        fig    = plot_zernike(coeffs, params, n_max, l_max, args.fit, args.text_threshold)
    else:
        l_max  = int(data['l_max'])
        params = parse_params_harmonic(data['params'])
        fig    = plot_harmonic(coeffs, params, l_max, args.fit, args.text_threshold)

    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Written → {args.out}')
    png_path = os.path.splitext(args.out)[0] + '.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Written → {png_path}')


if __name__ == '__main__':
    main()
