"""
NMR-only constrained fit: determine which harmonic correction modes the 4 NMR
probe measurements can constrain, and fit them.

The approach:
  1. Load the canonical TOSCA l_max=18 polynomial fit (fit to the full TOSCA grid).
  2. Evaluate it at the 4 NMR probe positions → get the baseline prediction.
  3. Compute residuals δBz = B_measured − B_TOSCA_fit.
  4. Build a correction design matrix for low-order modes (l_max=4, or axisymmetric only)
     at the 4 probe positions.
  5. Analyze via SVD: which linear combinations of modes are constrained?
  6. Fit the minimum-norm (or exact, for axisymmetric 4×4) correction.

Usage:
  python nmr_fit.py
  python nmr_fit.py --tosca tosca170812_full_coeffs_lmax18_all3.npz

The four physical combinations constrained by the NMR array:
  (A+E+C+D)/4  → l=1 scale (dominates, ~+2 mT)
  (A−E)/2      → φ-antisymmetric at r=2.9 m midplane (m=1 c or s)
  (D−C)/2      → z-antisymmetric at r=0.65 m endcap (missing-turn l-odd, m=0)
  (A+E)/2 −
  (C+D)/2      → z-radial shape (midplane vs. endcap, l=2 m=0 dominant)
"""

import numpy as np
import argparse
from harmonic_basis import build_design_matrix, eval_field, param_list

# ---------------------------------------------------------------------------
# NMR probe positions (CMS Cartesian, in cm) and 2017 measurements at 18164 A
# ---------------------------------------------------------------------------
NMR_PROBES = {
    'A': {'xyz_cm': (-206.345, -205.87,   -0.6),  'B_meas': 3.92165},
    'E': {'xyz_cm': (-206.345, +205.87,   +0.6),  'B_meas': 3.92118},
    'C': {'xyz_cm': ( +64.25,  +10.517, -283.5),  'B_meas': 3.65630},
    'D': {'xyz_cm': ( +64.25,  +10.517, +283.1),  'B_meas': 3.65990},
}
PROBE_LABELS = ['A', 'E', 'C', 'D']


def xyz_to_cyl(xyz_cm):
    """Convert CMS Cartesian (x,y,z) cm → cylindrical (r, phi, z) in cm."""
    x, y, z = xyz_cm
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi, z


def eval_at_nmr(coeffs, params, r_scale, probe_labels=PROBE_LABELS):
    """Evaluate saved polynomial fit at NMR probe positions."""
    r_arr, phi_arr, z_arr = [], [], []
    for lbl in probe_labels:
        r, phi, z = xyz_to_cyl(NMR_PROBES[lbl]['xyz_cm'])
        r_arr.append(r); phi_arr.append(phi); z_arr.append(z)
    r_arr   = np.array(r_arr)
    phi_arr = np.array(phi_arr)
    z_arr   = np.array(z_arr)
    result = eval_field(coeffs, params, r_arr, phi_arr, z_arr,
                        components=('Bz', 'Br', 'Bphi'), r_scale=r_scale)
    return result['Bz']


def load_tosca_fit(npz_path):
    """Load saved TOSCA polynomial fit."""
    d = np.load(npz_path, allow_pickle=True)
    coeffs   = d['coeffs']
    param_strs = d['params']
    r_scale  = float(d['r_scale'])
    # Reconstruct params as list of (l, m, cs) tuples
    params = []
    for s in param_strs:
        # format: l{l}_m{m}_{cs}
        parts = s.split('_')
        l  = int(parts[0][1:])
        m  = int(parts[1][1:])
        cs = parts[2]
        params.append((l, m, cs))
    return coeffs, params, r_scale


def build_correction_matrix(probe_labels, l_max, m_max=None, r_scale=None):
    """Build design matrix for correction modes at probe positions.

    Parameters
    ----------
    probe_labels : list of probe names to use
    l_max : maximum harmonic degree for correction
    m_max : maximum azimuthal order (default: same as l_max; set to 0 for axisymmetric only)
    r_scale : normalisation scale in cm (default: set from probe distances)

    Returns
    -------
    A : (n_probes, n_params) design matrix for Bz only
    params : list of (l, m, cs) — only modes with m <= m_max and l >= 1 (l=0 is gauge)
    r_scale : cm
    """
    r_arr, phi_arr, z_arr = [], [], []
    for lbl in probe_labels:
        r, phi, z = xyz_to_cyl(NMR_PROBES[lbl]['xyz_cm'])
        r_arr.append(r); phi_arr.append(phi); z_arr.append(z)
    r_arr   = np.array(r_arr)
    phi_arr = np.array(phi_arr)
    z_arr   = np.array(z_arr)

    if r_scale is None:
        R_sph = np.sqrt(r_arr**2 + z_arr**2)
        r_scale = float(np.max(R_sph))

    A_full, params_full, r_scale = build_design_matrix(
        r_arr, phi_arr, z_arr, l_max,
        components=('Bz',), r_scale=r_scale
    )
    # A_full has shape (n_probes * 1, n_params) = (n_probes, n_params) for Bz only

    # Filter to only l >= 1 and m <= m_max
    keep = []
    params_keep = []
    for i, (l, m, cs) in enumerate(params_full):
        if l == 0:
            continue  # gauge mode: gives B=0, undetermined
        if m_max is not None and m > m_max:
            continue
        keep.append(i)
        params_keep.append((l, m, cs))

    A_keep = A_full[:, keep]
    return A_keep, params_keep, r_scale


def svd_analysis(A, params, residuals_mT, labels=PROBE_LABELS):
    """SVD analysis of the correction design matrix.

    Prints singular values, identifies well-constrained vs. degenerate modes,
    and reports the minimum-norm correction.
    """
    n_probes, n_params = A.shape

    # Scale columns for fair SVD (avoids one large-norm column dominating)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    A_scaled  = A / col_norms[np.newaxis, :]

    U, sv, Vt = np.linalg.svd(A_scaled, full_matrices=False)

    print("\n=== SVD analysis ===")
    print(f"Design matrix: {n_probes} probes × {n_params} modes")
    print(f"Singular values: " + "  ".join(f"{s:.3e}" for s in sv))
    print()
    print("Measurement directions (left singular vectors = combinations of probes):")
    for k in range(len(sv)):
        combo = "  ".join(f"{labels[i]}:{U[i,k]:+.3f}" for i in range(n_probes))
        dom_param = np.argmax(np.abs(Vt[k]))
        l, m, cs = params[dom_param]
        print(f"  SV[{k}] = {sv[k]:.3e}  |  probe combo: {combo}")
        print(f"           dominant mode: l={l} m={m} {cs}  (V[{dom_param}]={Vt[k,dom_param]:.3f})")

    # Fit solution: least-squares for overdetermined (n_probes > n_params),
    #               minimum-norm for underdetermined (n_probes < n_params),
    #               exact for n_probes == n_params.
    b = np.array([residuals_mT[lbl] for lbl in labels]) * 1e-3  # mT → T
    # Use SVD pseudoinverse: c_scaled = V S^{-1} U^T b
    # This gives the minimum-norm least-squares solution in all cases.
    rhs = U.T @ b
    c_scaled = Vt.T @ (rhs / sv)  # shape (n_params,)
    coeffs_corr = c_scaled / col_norms

    print("\n=== Minimum-norm correction coefficients ===")
    print(f"{'Mode':20s}  {'coeff [T·cm]':>14s}  {'ΔBz at r=0,z=0 [mT]':>20s}")
    for i, (l, m, cs) in enumerate(params):
        c = coeffs_corr[i]
        if abs(c) > 1e-7:
            # Leading term: Bz(0,0) for axisymmetric modes (m=0) at origin = c * l * R^{l-1}/r_scale
            # At r=0, z=0: only l=1, m=0 gives nonzero Bz; higher l vanish at origin
            print(f"  l={l} m={m} {cs:3s}  {c:+14.6e}")

    # Predicted correction at NMR positions (residual of the fit)
    b_fit_mT = (A @ coeffs_corr) * 1e3
    print("\n=== Fit vs. measured residuals [mT] ===")
    print(f"{'Probe':6s}  {'δBz_meas':10s}  {'δBz_fit':10s}  {'leftover':10s}")
    for i, lbl in enumerate(labels):
        dm = residuals_mT[lbl]
        df = b_fit_mT[i]
        print(f"  {lbl:4s}  {dm:+10.3f}  {df:+10.3f}  {dm-df:+10.4f}")

    return coeffs_corr


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tosca', default='tosca170812_full_coeffs_lmax18_all3.npz',
                        help='Path to TOSCA l_max=18 fit .npz (default: tosca170812_full_coeffs_lmax18_all3.npz)')
    parser.add_argument('--lmax-corr', type=int, default=3,
                        help='Maximum l for NMR correction fit (default: 3, overdetermined by 1 with 4 probes)')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load the canonical TOSCA polynomial fit
    # -----------------------------------------------------------------------
    print(f"Loading TOSCA polynomial fit from {args.tosca}")
    coeffs_tosca, params_tosca, r_scale_tosca = load_tosca_fit(args.tosca)

    # -----------------------------------------------------------------------
    # Step 2: Evaluate TOSCA polynomial at NMR probe positions
    # -----------------------------------------------------------------------
    Bz_fit = eval_at_nmr(coeffs_tosca, params_tosca, r_scale_tosca)

    print("\n=== NMR probe comparison ===")
    print(f"{'Probe':6s}  {'r[m]':6s}  {'z[m]':7s}  {'B_meas[T]':11s}  "
          f"{'B_TOSCA_fit[T]':14s}  {'δBz[mT]':10s}")

    residuals_mT = {}
    for i, lbl in enumerate(PROBE_LABELS):
        r, phi, z = xyz_to_cyl(NMR_PROBES[lbl]['xyz_cm'])
        B_meas = NMR_PROBES[lbl]['B_meas']
        B_fit  = Bz_fit[i]
        delta  = (B_meas - B_fit) * 1e3
        residuals_mT[lbl] = delta
        print(f"  {lbl:4s}  {r/100:6.3f}  {z/100:7.3f}  {B_meas:11.5f}  "
              f"{B_fit:14.5f}  {delta:+10.3f}")

    # r_scale for correction: set from probe positions (encompass all probes)
    r_probe_max = max(
        np.sqrt(sum(xi**2 for xi in NMR_PROBES[lbl]['xyz_cm']))
        for lbl in PROBE_LABELS
    )
    r_scale_corr = float(r_probe_max) * 1.01  # 1% margin
    print(f"\nCorrection r_scale: {r_scale_corr:.1f} cm  (from probe positions; "
          f"TOSCA grid r_scale was {r_scale_tosca:.1f} cm)")

    # -----------------------------------------------------------------------
    # Step 3: Axisymmetric-only fit (m=0, l=1..l_max_corr)
    #   With l_max=3 and 4 probes: overdetermined by 1 → chi-square + residual
    #   With l_max=4 and 4 probes: exactly determined
    # -----------------------------------------------------------------------
    l_max_c = args.lmax_corr
    print(f"\n--- Axisymmetric (m=0) correction modes, l=1..{l_max_c} ---")
    A_axi, params_axi, r_scale_c = build_correction_matrix(
        PROBE_LABELS, l_max=l_max_c, m_max=0, r_scale=r_scale_corr
    )
    n_modes_axi = A_axi.shape[1]
    status = ('overdetermined' if A_axi.shape[0] > n_modes_axi else
              'exactly determined' if A_axi.shape[0] == n_modes_axi else 'underdetermined')
    print(f"Design matrix shape: {A_axi.shape} — {status}")
    if A_axi.shape[0] > n_modes_axi:
        print(f"({A_axi.shape[0]-n_modes_axi} residual DOF — chi-square goodness-of-fit available)")

    svd_analysis(A_axi, params_axi, residuals_mT)

    # -----------------------------------------------------------------------
    # Step 4: Include m=1 φ-asymmetric modes
    # -----------------------------------------------------------------------
    print(f"\n--- Adding m=1 correction modes, l=1..{l_max_c} (φ-asymmetric) ---")
    A_m1, params_m1, _ = build_correction_matrix(
        PROBE_LABELS, l_max=l_max_c, m_max=1, r_scale=r_scale_corr
    )
    print(f"Design matrix shape: {A_m1.shape} — "
          f"{'exactly determined' if A_m1.shape[0]==A_m1.shape[1] else 'underdetermined' if A_m1.shape[0]<A_m1.shape[1] else 'overdetermined'}")
    print("(4 probes, many modes → system underdetermined; minimum-norm solution reported)")

    svd_analysis(A_m1, params_m1, residuals_mT)

    # -----------------------------------------------------------------------
    # Step 5: Physical interpretation summary
    # -----------------------------------------------------------------------
    b_vals = np.array([residuals_mT[lbl] for lbl in PROBE_LABELS])  # mT
    A_sum  = b_vals.mean()
    AE_diff = (b_vals[0] - b_vals[1]) / 2
    DC_diff = (b_vals[3] - b_vals[2]) / 2
    AE_vs_CD = (b_vals[0] + b_vals[1]) / 2 - (b_vals[2] + b_vals[3]) / 2

    print("\n=== Four physical NMR constraints [mT] ===")
    print(f"  (A+E+C+D)/4 = {A_sum:+.3f}  → overall l=1 scale correction")
    print(f"  (A−E)/2     = {AE_diff:+.3f}  → φ-antisymmetric at r=2.9 m midplane (m=1 sin)")
    print(f"  (D−C)/2     = {DC_diff:+.3f}  → z-antisymmetric at r=0.65 m endcap (l-odd axisymmetric)")
    print(f"  (A+E)/2−(C+D)/2 = {AE_vs_CD:+.3f}  → z/radial shape (l=2 m=0 dominant)")
    print()
    print("Note: TOSCA l_max=18 polynomial residuals are what the NMR constrains.")
    print("The dominant correction is a uniform l=1 scale shift of +2 mT.")
    print("With 4 probes one can constrain 4 linear combinations of mode coefficients.")
    print("Track data (J/ψ and B→J/ψK) is needed to constrain spatial structure beyond l=2.")


if __name__ == '__main__':
    main()
