"""
Scan over the axial expansion origin z0 [cm] to study the effect of centring the
harmonic basis on the magnetic field centre rather than the geometric solenoid centre.

The CB-2 missing turn shifts the effective axial field centre by ~+13 mm toward +z.
Expanding the basis about z0 = z_centre means the field looks more z-symmetric
→ smaller odd-l coefficients, better conditioning at low l_max.

This script:
  1. Loads the TOSCA l_max=18 polynomial baseline (fitted with z0=0).
  2. For each z0 in a scan grid, recomputes the baseline prediction at NMR positions
     and refits the axisymmetric (m=0, l=1..l_max_corr) correction.
  3. Reports: fit chi-square, correction coefficients, and NMR predictions.

NOTE: The baseline itself was fitted with z0=0. For a fully consistent treatment,
the baseline should be re-fitted with the same z0. This script demonstrates the
effect on the NMR residual structure, which is independent of the baseline fitting
convention (the baseline prediction at the 4 NMR points changes only because
the basis re-parameterises the same polynomial).

Usage:
  python z0_scan.py
  python z0_scan.py --lmax-corr 3 --z0-max 2.5 --z0-step 0.25
"""

import numpy as np
import argparse
from harmonic_basis import build_design_matrix, eval_field, param_list

# NMR probe positions and 2017 measurements (same as nmr_fit.py)
NMR_PROBES = {
    'A': {'xyz_cm': (-206.345, -205.87,   -0.6),  'B_meas': 3.92165},
    'E': {'xyz_cm': (-206.345, +205.87,   +0.6),  'B_meas': 3.92118},
    'C': {'xyz_cm': ( +64.25,  +10.517, -283.5),  'B_meas': 3.65630},
    'D': {'xyz_cm': ( +64.25,  +10.517, +283.1),  'B_meas': 3.65990},
}
PROBE_LABELS = ['A', 'E', 'C', 'D']


def xyz_to_cyl(xyz_cm):
    x, y, z = xyz_cm
    return np.sqrt(x**2 + y**2), np.arctan2(y, x), z


def load_tosca_fit(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    coeffs = d['coeffs']
    param_strs = d['params']
    r_scale = float(d['r_scale'])
    z0_fit  = float(d['z0']) if 'z0' in d else 0.0
    params = []
    for s in param_strs:
        parts = s.split('_')
        params.append((int(parts[0][1:]), int(parts[1][1:]), parts[2]))
    return coeffs, params, r_scale, z0_fit


def nmr_baseline(coeffs, params, r_scale, z0_fit=0.0):
    """Evaluate TOSCA polynomial at NMR probe positions, return dict of Bz values."""
    r_arr, phi_arr, z_arr = zip(*(xyz_to_cyl(NMR_PROBES[l]['xyz_cm']) for l in PROBE_LABELS))
    res = eval_field(coeffs, params,
                     np.array(r_arr), np.array(phi_arr), np.array(z_arr),
                     components=('Bz',), r_scale=r_scale, z0=z0_fit)
    return {lbl: res['Bz'][i] for i, lbl in enumerate(PROBE_LABELS)}


def probe_r_scale(z0=0.0):
    """r_scale for the correction basis: from probe positions, shifted by z0."""
    R_max = 0.0
    for lbl in PROBE_LABELS:
        x, y, z = NMR_PROBES[lbl]['xyz_cm']
        R = np.sqrt(x**2 + y**2 + (z - z0)**2)
        R_max = max(R_max, R)
    return R_max * 1.01


def correction_fit(residuals_mT, l_max_corr, z0=0.0):
    """Fit axisymmetric (m=0) correction modes to the 4 NMR residuals.

    Returns
    -------
    coeffs : array  physical coefficients [T·cm]
    params : list of (l, m, cs)
    fit_mT : dict  probe label → fitted δBz [mT]
    rms    : float  RMS of (data - fit) [mT]  (0 if n_params >= n_probes)
    sv     : singular values of the design matrix
    """
    r_arr, phi_arr, z_arr = zip(*(xyz_to_cyl(NMR_PROBES[l]['xyz_cm']) for l in PROBE_LABELS))
    r_arr   = np.array(r_arr)
    phi_arr = np.array(phi_arr)
    z_arr   = np.array(z_arr)
    r_sc    = probe_r_scale(z0)

    # Build design matrix for Bz, axisymmetric (m=0), l=1..l_max_corr
    A_full, params_full, _ = build_design_matrix(
        r_arr, phi_arr, z_arr, l_max_corr,
        components=('Bz',), r_scale=r_sc, z0=z0
    )
    # Filter: l >= 1, m == 0 only
    keep = [i for i, (l, m, cs) in enumerate(params_full) if l >= 1 and m == 0]
    params_c = [params_full[i] for i in keep]
    A = A_full[:, keep]

    # Column-scale + SVD pseudoinverse
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled  = A / col_norms
    U, sv, Vt = np.linalg.svd(A_scaled, full_matrices=False)

    b = np.array([residuals_mT[lbl] for lbl in PROBE_LABELS]) * 1e-3  # T
    c_sc = Vt.T @ ((U.T @ b) / sv)
    coeffs = c_sc / col_norms

    b_fit = A @ coeffs
    fit_mT = {lbl: b_fit[i] * 1e3 for i, lbl in enumerate(PROBE_LABELS)}
    rms = float(np.sqrt(np.mean((b - b_fit)**2))) * 1e3  # mT

    return coeffs, params_c, fit_mT, rms, sv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tosca', default='tosca170812_full_coeffs_lmax18_all3.npz')
    parser.add_argument('--lmax-corr', type=int, default=3,
                        help='Max l for axisymmetric correction (default: 3)')
    parser.add_argument('--z0-max', type=float, default=2.5,
                        help='Maximum z0 scan value [cm] (default: 2.5 cm = 25 mm)')
    parser.add_argument('--z0-step', type=float, default=0.25,
                        help='Step size for z0 scan [cm] (default: 0.25 cm = 2.5 mm)')
    parser.add_argument('--grid', default=None,
                        help='If given, also run full-grid fit scan: path to field grid file '
                             '(e.g. ../MagneticField/Engine/test/field_tosca170812_full.txt)')
    parser.add_argument('--lmax-grid', type=int, default=6,
                        help='l_max for full-grid fit scan (default: 6)')
    parser.add_argument('--components', nargs='+', default=['Bz', 'Br'],
                        choices=['Bz', 'Br', 'Bphi'],
                        help='Field components for full-grid fit scan (default: Bz Br)')
    parser.add_argument('--tikhonov', type=float, default=0.0,
                        help='Tikhonov regularization for full-grid fit scan (default: 0)')
    parser.add_argument('--rmax-sphere', type=float, default=280.0, dest='rmax_sphere',
                        help='Spherical R cut [cm] for full-grid fit (default: 280)')
    args = parser.parse_args()

    coeffs_t, params_t, r_scale_t, z0_fit = load_tosca_fit(args.tosca)
    print(f"Loaded TOSCA baseline: {args.tosca}  (fitted with z0={z0_fit:.2f} cm)")

    # Baseline NMR predictions (z0=0 in the TOSCA polynomial)
    Bz_baseline = nmr_baseline(coeffs_t, params_t, r_scale_t, z0_fit)
    residuals_mT_0 = {lbl: (NMR_PROBES[lbl]['B_meas'] - Bz_baseline[lbl]) * 1e3
                      for lbl in PROBE_LABELS}

    print("\n=== NMR residuals used for correction (measured − TOSCA baseline) ===")
    print(f"{'Probe':6s}  {'δBz [mT]':>10s}")
    for lbl in PROBE_LABELS:
        print(f"  {lbl:4s}  {residuals_mT_0[lbl]:+10.3f}")

    z0_vals = np.arange(0.0, args.z0_max + args.z0_step * 0.5, args.z0_step)

    print(f"\n=== z0 scan: axisymmetric correction (m=0, l=1..{args.lmax_corr}) ===")
    print(f"{'z0[cm]':>8s}  {'RMS[mT]':>8s}  "
          + "  ".join(f"c_l{l:d}[T·cm]" for l in range(1, args.lmax_corr + 1))
          + f"  {'fit_A':>8s}  {'fit_E':>8s}  {'fit_C':>8s}  {'fit_D':>8s}  "
            f"{'leftA':>7s}  {'leftE':>7s}  {'leftC':>7s}  {'leftD':>7s}")

    results = []
    for z0 in z0_vals:
        coeffs_c, params_c, fit_mT, rms, sv = correction_fit(
            residuals_mT_0, args.lmax_corr, z0=z0
        )
        left = {lbl: residuals_mT_0[lbl] - fit_mT[lbl] for lbl in PROBE_LABELS}
        c_vals = [coeffs_c[i] for i in range(len(params_c))]
        c_str = "  ".join(f"{c:+10.4f}" for c in c_vals)
        print(f"{z0:8.2f}  {rms:8.4f}  {c_str}  "
              f"{fit_mT['A']:+8.3f}  {fit_mT['E']:+8.3f}  "
              f"{fit_mT['C']:+8.3f}  {fit_mT['D']:+8.3f}  "
              f"{left['A']:+7.4f}  {left['E']:+7.4f}  "
              f"{left['C']:+7.4f}  {left['D']:+7.4f}")
        results.append({'z0': z0, 'rms': rms, 'coeffs': coeffs_c,
                        'fit_mT': fit_mT, 'left': left, 'sv': sv})

    # Summary: highlight the z0 that minimises the RMS leftover
    best = min(results, key=lambda x: x['rms'])
    print(f"\nBest z0 (minimum correction RMS): {best['z0']:.2f} cm = {best['z0']*10:.1f} mm")
    print(f"  RMS leftover: {best['rms']:.4f} mT  (irreducible for m=0 fit)")

    # Physical combinations
    print("\n=== Physical combinations [mT] ===")
    print(f"  (A+E+C+D)/4 = {np.mean([residuals_mT_0[l] for l in PROBE_LABELS]):+.3f}  "
          f"[l=1 scale — well constrained by all z0]")
    print(f"  (A−E)/2     = {(residuals_mT_0['A']-residuals_mT_0['E'])/2:+.3f}  "
          f"[φ-antisymmetric — cannot be absorbed by axisymmetric modes regardless of z0]")
    print(f"  (D−C)/2     = {(residuals_mT_0['D']-residuals_mT_0['C'])/2:+.3f}  "
          f"[z-asymmetry — absorbed by l=3 m=0; sensitive to z0]")
    print(f"  (A+E)/2−(C+D)/2 = {(residuals_mT_0['A']+residuals_mT_0['E'])/2 - (residuals_mT_0['C']+residuals_mT_0['D'])/2:+.3f}  "
          f"[z/radial shape — absorbed by l=2 m=0; mildly sensitive to z0]")

    print("\nNote: the irreducible leftover for any axisymmetric fit is (A−E)/2 = "
          f"{(residuals_mT_0['A']-residuals_mT_0['E'])/2:+.3f} mT, "
          f"comparable to the ~0.4–0.6 mT probe z-position systematic.")

    if args.grid:
        z0_vals_grid = np.arange(0.0, args.z0_max + args.z0_step * 0.5, args.z0_step)
        grid_scan(args.grid, args.lmax_grid, args.components, args.tikhonov,
                  z0_vals_grid, rmax_sphere_cm=args.rmax_sphere)


def grid_scan(grid_path, l_max, components, tikhonov, z0_vals, rmax_sphere_cm=280.0):
    """For each z0, fit the full field grid and report RMS residuals per component.

    Returns list of dicts: z0, rms per component, condition number, odd-l power fraction.
    """
    from fit_field import load_grid, fit

    grid = load_grid(grid_path, rmax_sphere_cm=rmax_sphere_cm)
    n = len(grid['r_cm'])
    print(f"\nLoaded {n} grid points from {grid_path}")
    print(f"Grid scan: l_max={l_max}, components={components}, tikhonov={tikhonov:.1e}")
    print(f"\n{'z0[cm]':>8s}  {'z0[mm]':>7s}  "
          + "  ".join(f"RMS_{c}[mT]" for c in components)
          + "  odd_frac  cond_num")

    results = []
    for z0 in z0_vals:
        coeffs, params, info = fit(grid, l_max, components=tuple(components),
                                   tikhonov=tikhonov, z0=z0)
        rms_vals = [info['rms'].get(c, float('nan')) for c in components]

        # Fraction of l>1 variation power that is in odd-l modes.
        # Exclude l=0 (gauge) and l=1 (dominant uniform background ~3.8 T);
        # those dwarf everything else in raw c^2.
        # "odd_frac" answers: of the non-trivial variation modes, how much is z-antisymmetric?
        c_arr     = np.array(coeffs)
        odd_mask  = np.array([l % 2 == 1 and l > 1 for l, m, cs in params], dtype=float)
        bg_mask   = np.array([l > 1 for l, m, cs in params], dtype=float)
        bg_pow    = np.sum((c_arr * bg_mask)**2)
        odd_pow   = np.sum((c_arr * odd_mask)**2)
        odd_frac  = odd_pow / bg_pow if bg_pow > 0 else 0.0

        rms_str = "  ".join(f"{v*1e3:10.4f}" for v in rms_vals)
        print(f"{z0:8.2f}  {z0*10:7.1f}  {rms_str}  {odd_frac:9.4f}  {info['condition_number']:.2e}")
        results.append({'z0': z0, 'rms': rms_vals, 'odd_frac': odd_frac,
                        'cond': info['condition_number'], 'coeffs': coeffs, 'params': params})

    best = min(results, key=lambda x: x['rms'][0])
    print(f"\nBest z0 for RMS_{components[0]}: {best['z0']:.2f} cm = {best['z0']*10:.1f} mm "
          f"(RMS = {best['rms'][0]*1e3:.4f} mT)")

    return results


if __name__ == '__main__':
    main()
