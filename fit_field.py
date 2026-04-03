"""
Fit the Maroussov harmonic basis to a magnetic field grid dumped by CMSSW dumpField_cfg.py.

Usage:
  python fit_field.py --input field_tosca170812.txt --lmax 18 --components Bz Br
  python fit_field.py --input field_polyfit3d.txt   --lmax 6  --components Bz

Output:
  <input_stem>_coeffs_lmax<N>.npz  — numpy archive with:
    coeffs    : fitted coefficients, shape (n_params,)
    params    : parameter labels as strings "l{l}_m{m}_{cs}"
    residuals : per-point residuals for each fitted component
    rms       : RMS residuals per component

The grid format is the output of MagneticField/Engine/test/dumpField_cfg.py:
  # r[cm] phi[rad] z[cm] x[cm] y[cm] Bx[T] By[T] Bz[T]

Coordinates are converted to meters internally (matching BFit3D convention) but
all outputs label units clearly.

Notes on fitting strategy:
  - For a high-order fit to the TOSCA grid (l_max~18), use all grid points and
    --components Bz Br (or Bz Br Bphi) for best conditioning.
  - For a low-order residual fit (~l_max 6), restrict to the tracker volume
    (r < 115 cm, |z| < 280 cm) and use only grid points (not the NMR probe points).
  - The --rmax and --zmax flags select the fitting region.
"""

import numpy as np
import argparse
import os
from harmonic_basis import build_design_matrix, eval_field, param_list
from cylindrical_basis import (build_design_matrix_cyl, eval_field_cyl,
                                param_list_cyl, mode_label, parse_mode_label)


def load_grid(path, rmax_cm=None, zmax_cm=None, rmax_sphere_cm=None):
    """Load a field grid file produced by dumpField_cfg.py.

    Returns dict with keys:
      r_cm, phi, z_cm, x_cm, y_cm, Bx, By, Bz [T], Br [T], Bphi [T]

    Parameters
    ----------
    rmax_cm        : cylindrical r cut [cm]
    zmax_cm        : |z| cut [cm]
    rmax_sphere_cm : spherical R = sqrt(r^2+z^2) cut [cm] — use to stay inside
                     the current-free region (R < coil inner radius ~290 cm).
                     Essential for the high-order harmonic fit.
    """
    data = np.loadtxt(path, comments='#')
    # columns: r phi z x y Bx By Bz
    r_cm   = data[:, 0]
    phi    = data[:, 1]
    z_cm   = data[:, 2]
    x_cm   = data[:, 3]
    y_cm   = data[:, 4]
    Bx     = data[:, 5]
    By     = data[:, 6]
    Bz     = data[:, 7]

    # derived cylindrical components
    Br   = Bx * np.cos(phi) + By * np.sin(phi)
    Bphi = -Bx * np.sin(phi) + By * np.cos(phi)

    grid = {
        'r_cm': r_cm, 'phi': phi, 'z_cm': z_cm,
        'x_cm': x_cm, 'y_cm': y_cm,
        'Bx': Bx, 'By': By, 'Bz': Bz,
        'Br': Br, 'Bphi': Bphi,
    }

    # Apply spatial cuts
    mask = np.ones(len(r_cm), dtype=bool)
    if rmax_cm is not None:
        mask &= r_cm <= rmax_cm
    if zmax_cm is not None:
        mask &= np.abs(z_cm) <= zmax_cm
    if rmax_sphere_cm is not None:
        R_sph = np.sqrt(r_cm**2 + z_cm**2)
        mask &= R_sph <= rmax_sphere_cm

    # Remove points with zero/tiny field (outside parametrization validity)
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    mask &= Bmag > 0.1

    for k in grid:
        grid[k] = grid[k][mask]

    n_kept = mask.sum()
    n_total = len(mask)
    print(f"Loaded {n_total} points from {path}, using {n_kept} after cuts.")
    return grid


def fit_cylindrical(grid, n_max, m_max, L, components=('Bz',), tikhonov=0.0):
    """Fit the cylindrical Fourier-Bessel basis to the grid data.

    Parameters
    ----------
    grid       : dict from load_grid
    n_max      : int  maximum wavenumber index (k_n = n*pi/L)
    m_max      : int  maximum azimuthal order
    L          : float  Fourier half-period [cm]
    components : tuple of str  components to fit simultaneously
    tikhonov   : float  Tikhonov regularization strength λ

    Returns
    -------
    coeffs  : ndarray, shape (n_params,)
    modes   : list of (n, m, cphi, cz) tuples
    info    : dict with residuals, rms, condition_number, etc.
    """
    r   = grid['r_cm']
    phi = grid['phi']
    z   = grid['z_cm']

    n_modes = 1 + 2 * n_max * (1 + 2 * m_max)
    print(f"Building cylindrical design matrix: {len(r)} points, "
          f"n_max={n_max}, m_max={m_max}, L={L:.1f} cm, "
          f"components={components} → {len(r)*len(components)} rows × {n_modes} cols")

    A, modes = build_design_matrix_cyl(r, phi, z, n_max, m_max, L,
                                        components=components)

    # Column scaling
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled = A / col_norms[np.newaxis, :]

    comp_data = {'Bz': grid['Bz'], 'Br': grid['Br'], 'Bphi': grid['Bphi']}
    b = np.concatenate([comp_data[c] for c in components])

    if tikhonov > 0:
        lam_sqrt = np.sqrt(tikhonov)
        A_solve = np.vstack([A_scaled, lam_sqrt * np.eye(n_modes)])
        b_solve = np.concatenate([b, np.zeros(n_modes)])
        print(f"Solving with Tikhonov λ={tikhonov:.2e} (augmented shape {A_solve.shape})...")
    else:
        A_solve = A_scaled
        b_solve = b
        print(f"Solving least squares (matrix shape {A_scaled.shape})...")

    coeffs_scaled, _, rank, sv = np.linalg.lstsq(A_solve, b_solve, rcond=None)
    coeffs = coeffs_scaled / col_norms

    cond = sv[0] / sv[-1] if len(sv) > 0 and sv[-1] > 0 else float('nan')
    print(f"  Rank: {rank}/{n_modes}, condition number: {cond:.2e}")

    b_fit = A @ coeffs
    rms = {}
    n = len(r)
    for ic, comp in enumerate(components):
        res = b[ic*n:(ic+1)*n] - b_fit[ic*n:(ic+1)*n]
        rms[comp] = np.sqrt(np.mean(res**2))
        print(f"  RMS residual {comp}: {rms[comp]*1000:.4f} mT")

    info = {
        'residuals': b - b_fit,
        'rms': rms,
        'condition_number': cond,
        'rank': rank,
        'singular_values': sv,
        'n_points': n,
        'components': list(components),
        'basis': 'cylindrical',
        'n_max': n_max,
        'm_max': m_max,
        'L': L,
        'col_norms': col_norms,
        'tikhonov': tikhonov,
    }
    return coeffs, modes, info


def save_results_cyl(output_path, coeffs, modes, info):
    """Save cylindrical fit results to a .npz file."""
    param_strs = np.array([mode_label(n, m, cp, cz) for n, m, cp, cz in modes])
    np.savez(
        output_path,
        coeffs=coeffs,
        params=param_strs,
        basis='cylindrical',
        n_max=info['n_max'],
        m_max=info['m_max'],
        L=info['L'],
        col_norms=info['col_norms'],
        tikhonov=info['tikhonov'],
        rms_Bz=info['rms'].get('Bz', np.nan),
        rms_Br=info['rms'].get('Br', np.nan),
        rms_Bphi=info['rms'].get('Bphi', np.nan),
        condition_number=info['condition_number'],
        components=info['components'],
    )
    print(f"Saved coefficients to {output_path}.npz")


def fit(grid, l_max, components=('Bz',), tikhonov=0.0, z0=0.0, tikhonov_power=0.0):
    """Fit the harmonic basis to the grid data.

    Parameters
    ----------
    grid            : dict from load_grid
    l_max           : int  maximum degree
    components      : tuple of str  components to fit simultaneously
    tikhonov        : float  Tikhonov regularization strength λ (default 0 = no regularization).
    z0              : float  axial offset of expansion origin [cm] (default 0).
    tikhonov_power  : float  spectral smoothness power s (default 0 = standard Tikhonov).
                      The regularization penalty for mode (l,m) is scaled by [l(l+1)]^s,
                      so high-l modes are penalised more strongly than low-l modes.
                      This suppresses boundary oscillations while preserving tracker accuracy.
                      s=0: uniform penalty (standard Tikhonov).
                      s=1: penalise gradient (H^1 Sobolev norm on sphere).
                      s=2: penalise Laplacian squared (very strong smoothing).
                      Typical useful range: s=1..3 together with λ=1e-6.

    Returns
    -------
    coeffs  : ndarray, shape (n_params,)
    params  : list of (l, m, cs) tuples
    info    : dict with residuals, rms, condition_number, r_scale
    """
    r   = grid['r_cm']
    phi = grid['phi']
    z   = grid['z_cm']

    # r_scale: normalise R to [0,1] for numerical stability at high l
    # Use shifted z for the sphere radius calculation
    zp = z - z0
    r_scale = float(np.max(np.sqrt(r**2 + zp**2)))
    n_params = (l_max + 1)**2 - 1  # l=0 gauge mode excluded
    print(f"Building design matrix: {len(r)} points, l_max={l_max}, r_scale={r_scale:.1f} cm, "
          f"components={components} → {len(r)*len(components)} rows × {n_params} cols")

    A, params, r_scale = build_design_matrix(r, phi, z, l_max, components=components,
                                              r_scale=r_scale, z0=z0)

    # Column scaling: normalize each column to unit L2 norm.
    # Transforms c_physical = c_scaled / col_norms; greatly improves conditioning
    # when basis functions span many orders of magnitude (high-l harmonics).
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled = A / col_norms[np.newaxis, :]

    # Build RHS vector
    comp_data = {'Bz': grid['Bz'], 'Br': grid['Br'], 'Bphi': grid['Bphi']}
    b = np.concatenate([comp_data[c] for c in components])

    if tikhonov > 0:
        # Spectral Tikhonov: penalty for mode (l,m,cs) is λ * [l(l+1)]^s
        # With s=0 this is standard Tikhonov (uniform penalty).
        # With s>0 high-l modes are penalised more, enforcing smoothness.
        lam_sqrt = np.sqrt(tikhonov)
        if tikhonov_power > 0:
            weights = np.array([
                (l * (l + 1)) ** tikhonov_power if l > 0 else 1.0
                for l, m, cs in params
            ])
            # Normalise so the geometric mean weight = 1 (keeps λ interpretation stable)
            weights /= np.exp(np.mean(np.log(weights)))
            reg_diag = lam_sqrt * np.diag(weights)
            print(f"Solving with spectral Tikhonov λ={tikhonov:.2e}, s={tikhonov_power} "
                  f"(augmented shape ({A_scaled.shape[0]+n_params}, {n_params}))...")
        else:
            reg_diag = lam_sqrt * np.eye(n_params)
            print(f"Solving with Tikhonov λ={tikhonov:.2e} (augmented shape "
                  f"({A_scaled.shape[0]+n_params}, {n_params}))...")
        A_solve = np.vstack([A_scaled, reg_diag])
        b_solve = np.concatenate([b, np.zeros(n_params)])
    else:
        A_solve = A_scaled
        b_solve = b
        print(f"Solving least squares (matrix shape {A_scaled.shape})...")

    coeffs_scaled, _, rank, sv = np.linalg.lstsq(A_solve, b_solve, rcond=None)

    # Rescale back to physical coefficients
    coeffs = coeffs_scaled / col_norms

    cond = sv[0] / sv[-1] if len(sv) > 0 and sv[-1] > 0 else float('nan')
    print(f"  Rank: {rank}/{n_params}, condition number: {cond:.2e}")

    # Per-component RMS residuals (evaluated on original A, not augmented)
    b_fit = A @ coeffs
    rms = {}
    n = len(r)
    for ic, comp in enumerate(components):
        res = b[ic*n:(ic+1)*n] - b_fit[ic*n:(ic+1)*n]
        rms[comp] = np.sqrt(np.mean(res**2))
        print(f"  RMS residual {comp}: {rms[comp]*1000:.4f} mT")

    info = {
        'residuals': b - b_fit,
        'rms': rms,
        'condition_number': cond,
        'rank': rank,
        'singular_values': sv,
        'n_points': n,
        'components': list(components),
        'l_max': l_max,
        'r_scale': r_scale,
        'z0': z0,
        'col_norms': col_norms,
        'tikhonov': tikhonov,
        'tikhonov_power': tikhonov_power,
    }

    return coeffs, params, info


def save_results(output_path, coeffs, params, info):
    """Save fit results to a .npz file."""
    param_strs = np.array([f"l{l}_m{m}_{cs}" for l, m, cs in params])
    np.savez(
        output_path,
        coeffs=coeffs,
        params=param_strs,
        l_max=info['l_max'],
        r_scale=info['r_scale'],
        z0=info.get('z0', 0.0),
        col_norms=info['col_norms'],
        tikhonov=info['tikhonov'],
        tikhonov_power=info.get('tikhonov_power', 0.0),
        rms_Bz=info['rms'].get('Bz', np.nan),
        rms_Br=info['rms'].get('Br', np.nan),
        rms_Bphi=info['rms'].get('Bphi', np.nan),
        condition_number=info['condition_number'],
        components=info['components'],
        rmax_cm=info.get('rmax_cm', np.nan),
        zmax_cm=info.get('zmax_cm', np.nan),
        rmax_sphere_cm=info.get('rmax_sphere_cm', np.nan),
    )
    print(f"Saved coefficients to {output_path}.npz")


def compare_at_points(coeffs, params, points_cm, r_scale=1.0, labels=None):
    """Evaluate fitted field at specific points and print a comparison table.

    Parameters
    ----------
    coeffs, params : from fit()
    points_cm : list of (x_cm, y_cm, z_cm) tuples
    labels    : list of str, optional names for each point
    """
    print("\n--- Field evaluation at specific points ---")
    print(f"{'Label':12s}  {'r[cm]':8s}  {'phi[rad]':9s}  {'z[cm]':8s}  "
          f"{'Bz_fit[T]':10s}  {'Br_fit[T]':10s}")

    for i, (x, y, z) in enumerate(points_cm):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        label = labels[i] if labels else f"pt{i}"

        result = eval_field(coeffs, params, [r], [phi], [z], components=('Bz', 'Br', 'Bphi'), r_scale=r_scale)
        print(f"{label:12s}  {r:8.3f}  {phi:9.5f}  {z:8.3f}  "
              f"{result['Bz'][0]:10.6f}  {result['Br'][0]:10.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Input grid file from dumpField_cfg.py')
    parser.add_argument('--basis', default='spherical',
                        choices=['spherical', 'cylindrical'],
                        help='Basis type (default: spherical)')
    parser.add_argument('--lmax', type=int, default=18,
                        help='[spherical] Maximum harmonic degree (default: 18)')
    parser.add_argument('--nmax', type=int, default=10,
                        help='[cylindrical] Maximum wavenumber index n (default: 10)')
    parser.add_argument('--mmax', type=int, default=1,
                        help='[cylindrical] Maximum azimuthal order m (default: 1)')
    parser.add_argument('--period', type=float, default=None,
                        help='[cylindrical] Fourier half-period L [cm] '
                             '(default: auto = max |z| in grid + 10%%)')
    parser.add_argument('--components', nargs='+', default=['Bz'],
                        choices=['Bz', 'Br', 'Bphi'],
                        help='Field components to fit (default: Bz)')
    parser.add_argument('--rmax', type=float, default=None,
                        help='Maximum cylindrical r [cm] to include in fit (default: all)')
    parser.add_argument('--zmax', type=float, default=None,
                        help='Maximum |z| [cm] to include in fit (default: all)')
    parser.add_argument('--tikhonov', type=float, default=0.0,
                        help='Tikhonov regularization strength λ (default: 0 = none). '
                             'Use ~1e-6 for l_max=18 with the sphere320 grid.')
    parser.add_argument('--tikhonov-power', type=float, default=0.0,
                        dest='tikhonov_power',
                        help='Spectral smoothness power s for l-dependent regularization '
                             '(default: 0 = standard uniform Tikhonov). '
                             'Penalty for mode l scales as [l(l+1)]^s, suppressing '
                             'boundary oscillations without degrading tracker accuracy. '
                             'Typical range: s=1..3.')
    parser.add_argument('--rmax-sphere', type=float, default=None, dest='rmax_sphere',
                        help='Maximum spherical R=sqrt(r^2+z^2) [cm] to include in fit. '
                             'Use ~280 cm for the high-order harmonic fit to stay inside '
                             'the current-free solenoid bore (coil inner radius ~290 cm).')
    parser.add_argument('-o', '--output', default=None,
                        help='Output .npz file stem (default: derived from input)')
    parser.add_argument('--nmr', action='store_true',
                        help='Print field evaluation at NMR probe positions')
    parser.add_argument('--z0', type=float, default=0.0,
                        help='Axial offset of expansion origin [cm] (default: 0). '
                             'Set to the magnetic centre shift, e.g. +1.3 cm for CMS 3.8 T, '
                             'to reduce odd-l content and improve coefficient interpretability.')
    args = parser.parse_args()

    grid = load_grid(args.input, rmax_cm=args.rmax, zmax_cm=args.zmax,
                     rmax_sphere_cm=args.rmax_sphere)

    stem = args.output

    if args.basis == 'cylindrical':
        L = args.period
        if L is None:
            L = float(np.max(np.abs(grid['z_cm']))) * 1.05
            print(f"Auto period: L = {L:.1f} cm")
        coeffs, modes, info = fit_cylindrical(
            grid, args.nmax, args.mmax, L,
            components=tuple(args.components),
            tikhonov=args.tikhonov)
        if stem is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            stem = f"{base}_coeffs_cyl_nmax{args.nmax}_mmax{args.mmax}"
        save_results_cyl(stem, coeffs, modes, info)
    else:
        coeffs, params, info = fit(grid, args.lmax, components=tuple(args.components),
                                   tikhonov=args.tikhonov, z0=args.z0,
                                   tikhonov_power=args.tikhonov_power)
        info['rmax_cm'] = args.rmax if args.rmax is not None else np.nan
        info['zmax_cm'] = args.zmax if args.zmax is not None else np.nan
        info['rmax_sphere_cm'] = args.rmax_sphere if args.rmax_sphere is not None else np.nan
        if stem is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            stem = f"{base}_coeffs_lmax{args.lmax}"
        save_results(stem, coeffs, params, info)

    if args.nmr:
        # NMR probe positions in cm (CMS Cartesian)
        nmr_points = [
            (-206.345, -205.87, -0.6),   # A
            (-206.345, +205.87, +0.6),   # E
            (+64.25,   +10.517, -283.5), # C
            (+64.25,   +10.517, +283.1), # D
        ]
        compare_at_points(coeffs, params, nmr_points, r_scale=info['r_scale'],
                          labels=['A', 'E', 'C', 'D'])


if __name__ == '__main__':
    main()
