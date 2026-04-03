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
from cylindrical_bessel_basis import (build_design_matrix_cjb, eval_field_cjb,
                                       param_list_cjb,
                                       mode_label as mode_label_cjb,
                                       parse_mode_label as parse_mode_label_cjb)
import zernike_basis


# CMSSW field-map volume z-boundaries (from log_convert.txt, grid_170812_3_8t).
# Points within CMSSW_VOL_EXCL_HW cm of these boundaries have distorted field
# values due to bilinear-interpolation kinks and should be excluded from the fit.
CMSSW_VOL_BOUNDARIES_CM = [126.8, 142.3, 181.3]   # positive side; mirrored to ±
CMSSW_VOL_EXCL_HW = 12.0                            # half-width of exclusion zone [cm]


def parse_m_max_per_l(spec):
    """Parse a --m-max-per-l string into a {l: max_m} dict.

    Format: comma-separated "LRANGE:MMAX" tokens where LRANGE is either a
    single integer or "L1-L2" (inclusive).  MMAX=-1 means skip that l.

    Example: "6-11:2,12-15:1,16:0,17:-1,18:0"
    """
    result = {}
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        lrange, mmax = token.split(':')
        mmax = int(mmax)
        if '-' in lrange:
            l1, l2 = lrange.split('-')
            for l in range(int(l1), int(l2) + 1):
                result[l] = mmax
        else:
            result[int(lrange)] = mmax
    return result


def vol_boundary_mask(z_cm, boundaries=None, half_width=None):
    """Return boolean mask: True for points NOT near a CMSSW volume boundary.

    Parameters
    ----------
    z_cm       : array of z coordinates [cm]
    boundaries : list of positive z values [cm] (mirrored to ±); default CMSSW_VOL_BOUNDARIES_CM
    half_width : exclusion half-width [cm]; default CMSSW_VOL_EXCL_HW
    """
    if boundaries is None:
        boundaries = CMSSW_VOL_BOUNDARIES_CM
    if half_width is None:
        half_width = CMSSW_VOL_EXCL_HW
    keep = np.ones(len(z_cm), dtype=bool)
    for z0 in boundaries:
        keep &= np.abs(np.abs(z_cm) - z0) > half_width
    return keep


def load_grid(path, rmax_cm=None, zmax_cm=None, rmax_sphere_cm=None,
              rmin_sphere_cm=None, exclude_vol_boundaries=False):
    """Load a field grid file produced by dumpField_cfg.py.

    Returns dict with keys:
      r_cm, phi, z_cm, x_cm, y_cm, Bx, By, Bz [T], Br [T], Bphi [T]

    Parameters
    ----------
    rmax_cm               : cylindrical r cut [cm]
    zmax_cm               : |z| cut [cm]
    rmax_sphere_cm        : spherical R = sqrt(r^2+z^2) upper cut [cm]
    rmin_sphere_cm        : spherical R lower cut [cm] (removes origin and near-axis points)
    exclude_vol_boundaries: if True, remove points within CMSSW_VOL_EXCL_HW cm of
                            each CMSSW_VOL_BOUNDARIES_CM boundary (kink-affected points)
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
    if rmax_sphere_cm is not None or rmin_sphere_cm is not None:
        R_sph = np.sqrt(r_cm**2 + z_cm**2)
        if rmax_sphere_cm is not None:
            mask &= R_sph <= rmax_sphere_cm
        if rmin_sphere_cm is not None:
            mask &= R_sph >= rmin_sphere_cm
    if exclude_vol_boundaries:
        mask &= vol_boundary_mask(z_cm)

    # Remove points with zero/tiny field (outside parametrization validity)
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    mask &= Bmag > 0.1

    for k in grid:
        grid[k] = grid[k][mask]

    n_kept = mask.sum()
    n_total = len(mask)
    excl_str = " (vol boundaries excluded)" if exclude_vol_boundaries else ""
    print(f"Loaded {n_total} points from {path}, using {n_kept} after cuts{excl_str}.")
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
        basis_type='cylindrical',
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
        rmax_cm=info.get('rmax_cm', np.nan),
        zmax_cm=info.get('zmax_cm', np.nan),
        rmax_sphere_cm=info.get('rmax_sphere_cm', np.nan),
        rmin_sphere_cm=info.get('rmin_sphere_cm', np.nan),
        exclude_vol_boundaries=info.get('exclude_vol_boundaries', False),
    )
    print(f"Saved coefficients to {output_path}.npz")


def fit_bessel(grid, n_max, m_max, R=None, components=('Bz',), tikhonov=0.0):
    """Fit the J_m-Bessel + cosh/sinh basis to the grid data.

    Parameters
    ----------
    grid       : dict from load_grid
    n_max      : int    number of Bessel zeros per azimuthal order (1st..n_max-th zero of J_m)
    m_max      : int    maximum azimuthal order m
    R          : float  radial scale [cm]; Bessel zeros discretized as alpha_{m,n}/R.
                        Default: max(r_cm) in the fitting domain.
    components : tuple of str  components to fit simultaneously
    tikhonov   : float  Tikhonov regularization strength λ

    Returns
    -------
    coeffs, modes, info
    """
    r   = grid['r_cm']
    phi = grid['phi']
    z   = grid['z_cm']

    if R is None:
        R = float(np.max(r))
        print(f"Auto R (radial scale): {R:.1f} cm")

    n_modes = 1 + 2 * n_max * (1 + 2 * m_max)
    print(f"Building J-Bessel design matrix: {len(r)} points, "
          f"n_max={n_max}, m_max={m_max}, R={R:.1f} cm, "
          f"components={components} → {len(r)*len(components)} rows × {n_modes} cols")

    A, modes = build_design_matrix_cjb(r, phi, z, n_max, m_max, R, components=components)

    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled  = A / col_norms[np.newaxis, :]

    comp_data = {'Bz': grid['Bz'], 'Br': grid['Br'], 'Bphi': grid['Bphi']}
    b = np.concatenate([comp_data[c] for c in components])

    if tikhonov > 0:
        lam_sqrt = np.sqrt(tikhonov)
        A_solve  = np.vstack([A_scaled, lam_sqrt * np.eye(n_modes)])
        b_solve  = np.concatenate([b, np.zeros(n_modes)])
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
    rms   = {}
    n_pts = len(r)
    for ic, comp in enumerate(components):
        res = b[ic*n_pts:(ic+1)*n_pts] - b_fit[ic*n_pts:(ic+1)*n_pts]
        rms[comp] = np.sqrt(np.mean(res**2))
        print(f"  RMS residual {comp}: {rms[comp]*1000:.4f} mT")

    info = {
        'residuals': b - b_fit,
        'rms': rms,
        'condition_number': cond,
        'rank': rank,
        'singular_values': sv,
        'n_points': n_pts,
        'components': list(components),
        'basis': 'bessel',
        'n_max': n_max,
        'm_max': m_max,
        'R': R,
        'col_norms': col_norms,
        'tikhonov': tikhonov,
    }
    return coeffs, modes, info


def save_results_bessel(output_path, coeffs, modes, info):
    """Save J-Bessel fit results to a .npz file."""
    param_strs = np.array([mode_label_cjb(n, m, cp, cz) for n, m, cp, cz in modes])
    np.savez(
        output_path,
        coeffs=coeffs,
        params=param_strs,
        basis_type='bessel',
        n_max=info['n_max'],
        m_max=info['m_max'],
        R=info['R'],
        col_norms=info['col_norms'],
        tikhonov=info['tikhonov'],
        rms_Bz=info['rms'].get('Bz', np.nan),
        rms_Br=info['rms'].get('Br', np.nan),
        rms_Bphi=info['rms'].get('Bphi', np.nan),
        condition_number=info['condition_number'],
        components=info['components'],
        rmax_cm=info.get('rmax_cm', np.nan),
        zmax_cm=info.get('zmax_cm', np.nan),
        rmax_sphere_cm=info.get('rmax_sphere_cm', np.nan),
        rmin_sphere_cm=info.get('rmin_sphere_cm', np.nan),
        exclude_vol_boundaries=info.get('exclude_vol_boundaries', False),
    )
    print(f"Saved coefficients to {output_path}.npz")


def gl_point_weights(r_cm, z_cm, l_max, z0=0.0):
    """Gauss-Legendre correction weights for a uniform-z cylindrical grid.

    A uniform-z grid samples cos θ = z/R uniformly on any sphere of fixed R,
    which is the worst case for Legendre polynomial fitting (analogous to
    equally-spaced nodes in the Runge phenomenon).  Gauss-Legendre nodes in
    cos θ, which cluster near ±1, are optimal.  These weights correct for the
    mismatch by upweighting polar points (|cos θ| → 1) and downweighting
    equatorial points (cos θ → 0), making the discrete inner product closer to
    the GL quadrature rule for l_max+1 nodes.

    Weight: w_i = w_GL(cos θ_i) where w_GL is interpolated from the n=l_max+1
    Gauss-Legendre quadrature weights.  Weights are normalised to mean 1.
    """
    nodes, weights = np.polynomial.legendre.leggauss(l_max + 1)
    zp = z_cm - z0
    R = np.sqrt(r_cm**2 + zp**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = np.where(R > 0, zp / R, np.sign(zp))
    w = np.interp(cos_theta, nodes, weights)
    w = np.maximum(w, 1e-6 * w.max())   # floor to avoid zero weights at r=0 axis
    w /= w.mean()                         # normalise: mean weight = 1
    return w


def fit(grid, l_max, components=('Bz',), tikhonov=0.0, z0=0.0, tikhonov_power=0.0,
        gl_weights=False, l_max_phi=None, n_max_sum=None, m_max_per_l=None, r_scale=None):
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
    if r_scale is None:
        r_scale = float(np.max(np.sqrt(r**2 + zp**2)))
    from harmonic_basis import param_list as _param_list
    n_params = len(_param_list(l_max, l_max_phi=l_max_phi, n_max_sum=n_max_sum,
                               m_max_per_l=m_max_per_l))
    if m_max_per_l is not None:
        lmax_str = f"l_max={l_max}, custom m_max_per_l ({n_params} params)"
    elif n_max_sum is not None:
        lmax_str = f"diamond n_max_sum={n_max_sum}"
    elif l_max_phi is not None and l_max_phi != l_max:
        lmax_str = f"l_max={l_max}, l_max_phi={l_max_phi}"
    else:
        lmax_str = f"l_max={l_max}"
    print(f"Building design matrix: {len(r)} points, {lmax_str}, r_scale={r_scale:.1f} cm, "
          f"components={components} → {len(r)*len(components)} rows × {n_params} cols")

    A, params, r_scale = build_design_matrix(r, phi, z, l_max, components=components,
                                              r_scale=r_scale, z0=z0,
                                              l_max_phi=l_max_phi, n_max_sum=n_max_sum,
                                              m_max_per_l=m_max_per_l)

    # Optional Gauss-Legendre correction weights (see gl_point_weights)
    if gl_weights:
        w = gl_point_weights(r, z, l_max, z0=z0)
        # Tile weights for all components (each point appears once per component)
        w_full = np.tile(np.sqrt(w), len(components))
        print(f"  GL weights: min={w.min():.3f} max={w.max():.3f} "
              f"(upweight |cosθ|→1, downweight equator)")
    else:
        w_full = None

    # Column scaling: normalize each column to unit L2 norm.
    # Transforms c_physical = c_scaled / col_norms; greatly improves conditioning
    # when basis functions span many orders of magnitude (high-l harmonics).
    if w_full is not None:
        col_norms = np.linalg.norm(A * w_full[:, np.newaxis], axis=0)
    else:
        col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled = A / col_norms[np.newaxis, :]

    # Build RHS vector (keep unweighted copy for residual evaluation)
    comp_data = {'Bz': grid['Bz'], 'Br': grid['Br'], 'Bphi': grid['Bphi']}
    b = np.concatenate([comp_data[c] for c in components])
    b_unweighted = b.copy()

    # Apply GL row weights to A_scaled and b (weighted solve only; RMS uses b_unweighted)
    if w_full is not None:
        A_scaled = A_scaled * w_full[:, np.newaxis]
        b = b * w_full

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

    # Per-component RMS residuals (always on physical/unweighted quantities)
    b_fit = A @ coeffs
    rms = {}
    n = len(r)
    for ic, comp in enumerate(components):
        res = b_unweighted[ic*n:(ic+1)*n] - b_fit[ic*n:(ic+1)*n]
        rms[comp] = np.sqrt(np.mean(res**2))
        print(f"  RMS residual {comp}: {rms[comp]*1000:.4f} mT")

    info = {
        'residuals': b_unweighted - b_fit,
        'rms': rms,
        'condition_number': cond,
        'rank': rank,
        'singular_values': sv,
        'n_points': n,
        'components': list(components),
        'l_max': l_max,
        'l_max_phi': l_max_phi,
        'n_max_sum': n_max_sum,
        'm_max_per_l': m_max_per_l,
        'r_scale': r_scale,
        'z0': z0,
        'col_norms': col_norms,
        'tikhonov': tikhonov,
        'tikhonov_power': tikhonov_power,
    }

    return coeffs, params, info


def fit_zernike(grid, n_max, l_max=None, components=('Bz',),
                tikhonov=0.0, tikhonov_power=0.0, z0=0.0):
    """Fit the 3D Zernike scalar-potential basis to the grid data.

    Parameters
    ----------
    grid       : dict from load_grid
    n_max      : int  maximum total Zernike degree  (radial resolution)
    l_max      : int  maximum angular degree, i.e. max m = l_max (default = n_max)
    components : tuple of str  ('Bz', 'Br', 'Bphi')
    tikhonov   : float  regularization strength λ (0 = none)
    tikhonov_power : float  spectral power s; penalty scales as [n(n+1)]^s
    z0         : float  axial origin offset [cm]

    Returns
    -------
    coeffs, params, info  (same structure as fit())
    """
    if l_max is None:
        l_max = n_max

    r   = grid['r_cm']
    phi = grid['phi']
    z   = grid['z_cm']

    zp = z - z0
    r_scale = float(np.max(np.sqrt(r**2 + zp**2)))

    params  = zernike_basis.param_list(n_max, l_max)
    n_par   = len(params)

    print(f"Building Zernike design matrix: {len(r)} points, "
          f"n_max={n_max}, l_max={l_max}, r_scale={r_scale:.1f} cm, "
          f"components={components} → {len(r)*len(components)} rows × {n_par} cols")

    A, params, r_scale = zernike_basis.build_design_matrix(
        r, phi, z, n_max, l_max,
        components=components, r_scale=r_scale, z0=z0)

    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled  = A / col_norms[np.newaxis, :]

    comp_data   = {'Bz': grid['Bz'], 'Br': grid['Br'], 'Bphi': grid['Bphi']}
    b           = np.concatenate([comp_data[c] for c in components])
    b_unweighted = b.copy()

    if tikhonov > 0:
        lam_sqrt = np.sqrt(tikhonov)
        if tikhonov_power > 0:
            weights = np.array([
                (nv * (nv + 1)) ** tikhonov_power if nv > 0 else 1.0
                for nv, lv, mv, csv in params
            ])
            weights /= np.exp(np.mean(np.log(weights)))
            reg_diag = lam_sqrt * np.diag(weights)
        else:
            reg_diag = lam_sqrt * np.eye(n_par)
        A_solve = np.vstack([A_scaled, reg_diag])
        b_solve = np.concatenate([b, np.zeros(n_par)])
        print(f"Solving Zernike with Tikhonov λ={tikhonov:.2e}, s={tikhonov_power} ...")
    else:
        A_solve = A_scaled
        b_solve = b
        print(f"Solving Zernike least squares (shape {A_scaled.shape}) ...")

    coeffs_scaled, _, rank, sv = np.linalg.lstsq(A_solve, b_solve, rcond=None)
    coeffs = coeffs_scaled / col_norms

    cond = sv[0] / sv[-1] if len(sv) > 0 and sv[-1] > 0 else float('nan')
    print(f"  Rank: {rank}/{n_par}, condition number: {cond:.2e}")

    b_fit = A @ coeffs
    rms   = {}
    n_pts = len(r)
    for ic, comp in enumerate(components):
        res = b_unweighted[ic*n_pts:(ic+1)*n_pts] - b_fit[ic*n_pts:(ic+1)*n_pts]
        rms[comp] = float(np.sqrt(np.mean(res**2)))
        print(f"  RMS residual {comp}: {rms[comp]*1000:.4f} mT")

    info = {
        'residuals': b_unweighted - b_fit,
        'rms': rms,
        'condition_number': cond,
        'rank': rank,
        'singular_values': sv,
        'n_points': n_pts,
        'components': list(components),
        'n_max': n_max,
        'l_max': l_max,
        'r_scale': r_scale,
        'z0': z0,
        'col_norms': col_norms,
        'tikhonov': tikhonov,
        'tikhonov_power': tikhonov_power,
        'basis_type': 'zernike',
    }
    return coeffs, params, info


def save_results_zernike(output_path, coeffs, params, info):
    """Save Zernike fit results to a .npz file."""
    param_strs = np.array([f"n{n}_l{l}_m{m}_{cs}" for n, l, m, cs in params])
    np.savez(
        output_path,
        coeffs=coeffs,
        params=param_strs,
        basis_type='zernike',
        n_max=info['n_max'],
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
        rmin_sphere_cm=info.get('rmin_sphere_cm', np.nan),
        exclude_vol_boundaries=info.get('exclude_vol_boundaries', False),
    )
    print(f"Saved Zernike coefficients to {output_path}.npz")


def save_results(output_path, coeffs, params, info):
    """Save fit results to a .npz file."""
    param_strs = np.array([f"l{l}_m{m}_{cs}" for l, m, cs in params])
    lmp  = info.get('l_max_phi')
    nms  = info.get('n_max_sum')
    mmpl = info.get('m_max_per_l')
    # Serialise m_max_per_l as a string so it round-trips through npz
    mmpl_str = (','.join(f"{l}:{m}" for l, m in sorted(mmpl.items()))
                if mmpl is not None else None)
    np.savez(
        output_path,
        coeffs=coeffs,
        params=param_strs,
        basis_type='harmonic',
        l_max=info['l_max'],
        **({'l_max_phi': lmp} if lmp is not None else {}),
        **({'n_max_sum': nms} if nms is not None else {}),
        **({'m_max_per_l_str': mmpl_str} if mmpl_str is not None else {}),
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
        rmin_sphere_cm=info.get('rmin_sphere_cm', np.nan),
        exclude_vol_boundaries=info.get('exclude_vol_boundaries', False),
        gl_weights=info.get('gl_weights', False),
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
                        choices=['spherical', 'cylindrical', 'bessel', 'zernike'],
                        help='Basis type (default: spherical). '
                             '"bessel" = J_m Bessel + cosh/sinh z (hyperbolic cylindrical)')
    parser.add_argument('--lmax', type=int, default=18,
                        help='[spherical] Maximum harmonic degree (default: 18)')
    parser.add_argument('--lmax-phi', type=int, default=None, dest='lmax_phi',
                        help='[spherical] Maximum degree for m>0 (phi-dependent) modes. '
                             'If less than --lmax, gives an asymmetric truncation with high '
                             'theta-resolution (m=0 up to --lmax) and limited phi-resolution '
                             '(m>0 up to --lmax-phi). Default: same as --lmax (standard triangular).')
    parser.add_argument('--m-max-per-l', type=str, default=None, dest='m_max_per_l',
                        help='[spherical] Custom per-l maximum m envelope. '
                             'Format: comma-separated "LRANGE:MMAX" tokens where LRANGE '
                             'is a single integer or "L1-L2" (inclusive), and MMAX=-1 skips '
                             'that l entirely. Overrides --lmax-phi and --nmax-sum when set. '
                             'Example: "6-11:2,12-15:1,16:0,17:-1,18:0"')
    parser.add_argument('--nmax-sum', type=int, default=None, dest='nmax_sum',
                        help='[spherical] Diamond truncation: include (l,m) only if l+m <= nmax_sum. '
                             'm=0 modes go up to l=nmax_sum; high-m modes are progressively '
                             'cut off. Overrides --lmax and --lmax-phi when set.')
    parser.add_argument('--nmax', type=int, default=10,
                        help='[cylindrical] Maximum wavenumber index n (default: 10)')
    parser.add_argument('--mmax', type=int, default=1,
                        help='[cylindrical] Maximum azimuthal order m (default: 1)')
    parser.add_argument('--period', type=float, default=None,
                        help='[cylindrical] Fourier half-period L [cm] '
                             '(default: auto = max |z| in grid + 10%%)')
    parser.add_argument('--rscale', type=float, default=None,
                        help='[bessel] Radial scale R [cm] for Bessel zero discretisation '
                             'k_{m,n} = alpha_{m,n}/R  (default: auto = max r in grid)')
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
    parser.add_argument('--rmin-sphere', type=float, default=None, dest='rmin_sphere',
                        help='Minimum spherical R=sqrt(r^2+z^2) [cm] to include in fit. '
                             'Removes the origin and near-axis points where some bases '
                             '(e.g. Zernike) have numerical singularities. '
                             'A value of ~1 cm excludes only the r=0,z=0 grid point.')
    parser.add_argument('-o', '--output', default=None,
                        help='Output .npz file stem (default: derived from input)')
    parser.add_argument('--nmr', action='store_true',
                        help='Print field evaluation at NMR probe positions')
    parser.add_argument('--z0', type=float, default=0.0,
                        help='Axial offset of expansion origin [cm] (default: 0). '
                             'Set to the magnetic centre shift, e.g. +1.3 cm for CMS 3.8 T, '
                             'to reduce odd-l content and improve coefficient interpretability.')
    parser.add_argument('--gl-weights', action='store_true', dest='gl_weights',
                        help='Apply Gauss-Legendre correction weights to data points. '
                             'Upweights polar regions (|cos θ| → 1) and downweights the '
                             'equatorial band to correct for uniform-z oversampling of '
                             'the equator. Tests whether Runge oscillations at the sphere '
                             'boundary are sampling-driven.')
    parser.add_argument('--exclude-vol-boundaries', action='store_true',
                        dest='exclude_vol_boundaries',
                        help='Exclude points within %(default)s cm of the known CMSSW field-map '
                             'volume z-boundaries (±126.8, ±142.3, ±181.3 cm). '
                             'These points have bilinear-interpolation kink artifacts '
                             'at the 0.07–0.10 mT level.')
    args = parser.parse_args()

    grid = load_grid(args.input, rmax_cm=args.rmax, zmax_cm=args.zmax,
                     rmax_sphere_cm=args.rmax_sphere, rmin_sphere_cm=args.rmin_sphere,
                     exclude_vol_boundaries=args.exclude_vol_boundaries)

    stem = args.output

    if args.basis == 'zernike':
        l_max_zk = args.lmax if args.lmax != 18 else None   # use None (=n_max) if user left default
        # allow --lmax to set angular cap; if user explicitly passed it use it
        coeffs, params, info = fit_zernike(
            grid, args.nmax,
            l_max=args.lmax,
            components=tuple(args.components),
            tikhonov=args.tikhonov,
            tikhonov_power=args.tikhonov_power,
            z0=args.z0)
        info['rmax_cm']            = args.rmax         if args.rmax         is not None else np.nan
        info['zmax_cm']            = args.zmax         if args.zmax         is not None else np.nan
        info['rmax_sphere_cm']     = args.rmax_sphere  if args.rmax_sphere  is not None else np.nan
        info['rmin_sphere_cm']     = args.rmin_sphere  if args.rmin_sphere  is not None else np.nan
        info['exclude_vol_boundaries'] = args.exclude_vol_boundaries
        if stem is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            tag  = '_excl' if args.exclude_vol_boundaries else ''
            stem = f"{base}_zernike_nmax{args.nmax}_lmax{args.lmax}{tag}"
        save_results_zernike(stem, coeffs, params, info)
        if args.nmr:
            nmr_points = [
                (-206.345, -205.87, -0.6),
                (-206.345, +205.87, +0.6),
                (+64.25,   +10.517, -283.5),
                (+64.25,   +10.517, +283.1),
            ]
            from zernike_basis import eval_field as zeval
            print("\n--- Field evaluation at NMR probe positions ---")
            for label, (x, y, zv) in zip(['A','E','C','D'], nmr_points):
                rc = np.sqrt(x**2 + y**2); phic = np.arctan2(y, x)
                res = zeval(coeffs, params, [rc], [phic], [zv],
                            components=('Bz','Br'), r_scale=info['r_scale'], z0=args.z0)
                print(f"  {label}: Bz={res['Bz'][0]:.6f}  Br={res['Br'][0]:.6f}")
    elif args.basis == 'cylindrical':
        L = args.period
        if L is None:
            L = float(np.max(np.abs(grid['z_cm']))) * 1.05
            print(f"Auto period: L = {L:.1f} cm")
        coeffs, modes, info = fit_cylindrical(
            grid, args.nmax, args.mmax, L,
            components=tuple(args.components),
            tikhonov=args.tikhonov)
        info['rmax_cm']            = args.rmax        if args.rmax        is not None else np.nan
        info['zmax_cm']            = args.zmax        if args.zmax        is not None else np.nan
        info['rmax_sphere_cm']     = args.rmax_sphere if args.rmax_sphere is not None else np.nan
        info['rmin_sphere_cm']     = args.rmin_sphere if args.rmin_sphere is not None else np.nan
        info['exclude_vol_boundaries'] = args.exclude_vol_boundaries
        if stem is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            stem = f"{base}_coeffs_cyl_nmax{args.nmax}_mmax{args.mmax}"
        save_results_cyl(stem, coeffs, modes, info)
    elif args.basis == 'bessel':
        coeffs, modes, info = fit_bessel(
            grid, args.nmax, args.mmax, R=args.rscale,
            components=tuple(args.components),
            tikhonov=args.tikhonov)
        info['rmax_cm']            = args.rmax        if args.rmax        is not None else np.nan
        info['zmax_cm']            = args.zmax        if args.zmax        is not None else np.nan
        info['rmax_sphere_cm']     = args.rmax_sphere if args.rmax_sphere is not None else np.nan
        info['rmin_sphere_cm']     = args.rmin_sphere if args.rmin_sphere is not None else np.nan
        info['exclude_vol_boundaries'] = args.exclude_vol_boundaries
        if stem is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            stem = f"data/fitresults/{base}_coeffs_bessel_nmax{args.nmax}_mmax{args.mmax}"
        save_results_bessel(stem, coeffs, modes, info)
    else:
        m_max_per_l = parse_m_max_per_l(args.m_max_per_l) if args.m_max_per_l else None
        l_max_fit = args.nmax_sum if args.nmax_sum is not None else args.lmax
        coeffs, params, info = fit(grid, l_max_fit, components=tuple(args.components),
                                   tikhonov=args.tikhonov, z0=args.z0,
                                   tikhonov_power=args.tikhonov_power,
                                   gl_weights=args.gl_weights,
                                   l_max_phi=args.lmax_phi,
                                   n_max_sum=args.nmax_sum,
                                   m_max_per_l=m_max_per_l,
                                   r_scale=args.rscale)
        info['rmax_cm'] = args.rmax if args.rmax is not None else np.nan
        info['zmax_cm'] = args.zmax if args.zmax is not None else np.nan
        info['rmax_sphere_cm'] = args.rmax_sphere if args.rmax_sphere is not None else np.nan
        info['rmin_sphere_cm'] = args.rmin_sphere if args.rmin_sphere is not None else np.nan
        info['exclude_vol_boundaries'] = args.exclude_vol_boundaries
        info['gl_weights'] = args.gl_weights
        if stem is None:
            base = os.path.splitext(os.path.basename(args.input))[0]
            tag = '_excl' if args.exclude_vol_boundaries else ''
            tag += '_gl' if args.gl_weights else ''
            if args.m_max_per_l is not None:
                tag += '_custom'
            elif args.nmax_sum is not None:
                tag += f'_diamond{args.nmax_sum}'
            elif args.lmax_phi is not None and args.lmax_phi != args.lmax:
                tag += f'_lphi{args.lmax_phi}'
            stem = f"{base}_coeffs_lmax{l_max_fit}{tag}"
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
