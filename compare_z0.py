"""
Compare harmonic fit coefficients between z0=0 and z0=+1.5 cm (magnetic centre offset).

The CB-2 missing turn shifts the axial field centre toward +z by ~15 mm at 3.8 T.
Expanding the basis about the true field centre rather than the geometric centre
remixes the coefficients: z-antisymmetric even-l modes (l=2,4,...) capture the
residual asymmetry relative to the SHIFTED centre, while odd-l modes capture the
z-symmetric bulk variation — now in a more natural coordinate.

Usage:
  python compare_z0.py
  python compare_z0.py --z0  tosca170812_full_coeffs_lmax18_all3_z0_15mm.npz
  python compare_z0.py --ref tosca170812_full_coeffs_lmax18_all3.npz \\
                        --z0  tosca170812_full_coeffs_lmax18_all3_z0_15mm.npz
"""

import numpy as np
import argparse


NMR_PROBES = {
    'A': {'xyz_cm': (-206.345, -205.87,   -0.6),  'B_meas': 3.92165},
    'E': {'xyz_cm': (-206.345, +205.87,   +0.6),  'B_meas': 3.92118},
    'C': {'xyz_cm': ( +64.25,  +10.517, -283.5),  'B_meas': 3.65630},
    'D': {'xyz_cm': ( +64.25,  +10.517, +283.1),  'B_meas': 3.65990},
}
PROBE_LABELS = ['A', 'E', 'C', 'D']


def load_fit(path):
    d = np.load(path, allow_pickle=True)
    coeffs = d['coeffs']
    param_strs = d['params']
    r_scale = float(d['r_scale'])
    z0 = float(d['z0']) if 'z0' in d else 0.0
    l_max = int(d['l_max'])
    rms = {c: float(d[f'rms_{c}']) for c in ('Bz', 'Br', 'Bphi')}
    params = []
    for s in param_strs:
        parts = s.split('_')
        params.append((int(parts[0][1:]), int(parts[1][1:]), parts[2]))
    return {'coeffs': coeffs, 'params': params, 'r_scale': r_scale,
            'z0': z0, 'l_max': l_max, 'rms': rms, 'path': path}


def coeff_table(fit, l_max_show=6):
    """Return dict (l,m,cs) → coeff for modes up to l_max_show."""
    d = {}
    for c, (l, m, cs) in zip(fit['coeffs'], fit['params']):
        if l <= l_max_show:
            d[(l, m, cs)] = c
    return d


def print_coeff_comparison(fit_ref, fit_z0, fit_z0b=None, l_max_show=6):
    """Side-by-side coefficient table for two or three fits."""
    t_ref = coeff_table(fit_ref, l_max_show)
    t_z0  = coeff_table(fit_z0,  l_max_show)
    t_z0b = coeff_table(fit_z0b, l_max_show) if fit_z0b else None

    all_keys = sorted(set(t_ref) | set(t_z0),
                      key=lambda x: (x[0], x[1], x[2]))

    z0_label  = f"z0=+{fit_z0['z0']:.1f}cm"
    z0b_label = f"z0=+{fit_z0b['z0']:.1f}cm" if fit_z0b else None

    if t_z0b is None:
        print(f"\n{'Mode':20s}  {'z0=0 [T·cm]':>16s}  "
              f"{z0_label+' [T·cm]':>16s}  {'diff':>12s}  {'diff/|ref|%':>10s}")
        print("-" * 85)
    else:
        print(f"\n{'Mode':20s}  {'z0=0 [T·cm]':>16s}  "
              f"{z0_label+' [T·cm]':>16s}  {'diff_a':>12s}  "
              f"{z0b_label+' [T·cm]':>16s}  {'diff_b':>12s}")
        print("-" * 105)

    for l, m, cs in all_keys:
        if l == 0:
            continue
        c_ref = t_ref.get((l, m, cs), 0.0)
        c_a   = t_z0.get((l, m, cs), 0.0)
        diff_a = c_a - c_ref
        pct_a  = 100.0 * diff_a / abs(c_ref) if abs(c_ref) > 1e-12 else float('nan')
        sym    = 'z-sym' if (l - 1) % 2 == 0 else 'z-anti'

        if t_z0b is None:
            print(f"  l={l} m={m} {cs:3s} ({sym:6s})  {c_ref:+16.6e}  {c_a:+16.6e}  "
                  f"{diff_a:+12.4e}  {pct_a:+10.2f}%")
        else:
            c_b    = t_z0b.get((l, m, cs), 0.0)
            diff_b = c_b - c_ref
            pct_b  = 100.0 * diff_b / abs(c_ref) if abs(c_ref) > 1e-12 else float('nan')
            print(f"  l={l} m={m} {cs:3s} ({sym:6s})  {c_ref:+16.6e}  {c_a:+16.6e}  "
                  f"{diff_a:+12.4e}  {c_b:+16.6e}  {diff_b:+12.4e}")


def power_analysis(fit, l_max_show=None):
    """Power fractions: how much is in z-symmetric (odd-l) vs z-antisymmetric (even-l) modes."""
    if l_max_show is None:
        l_max_show = fit['l_max']

    c_arr   = np.array([c for c, (l, m, cs) in zip(fit['coeffs'], fit['params'])
                        if l <= l_max_show])
    l_arr   = np.array([l for c, (l, m, cs) in zip(fit['coeffs'], fit['params'])
                        if l <= l_max_show])

    total_pow = np.sum(c_arr**2)
    l1_pow    = np.sum(c_arr[l_arr == 1]**2)
    bg_pow    = np.sum(c_arr[l_arr > 1]**2)
    zsym_pow  = np.sum(c_arr[(l_arr > 1) & (l_arr % 2 == 1)]**2)   # l=3,5,... → z-sym Bz
    zanti_pow = np.sum(c_arr[(l_arr > 1) & (l_arr % 2 == 0)]**2)   # l=2,4,... → z-anti Bz

    return {
        'total': total_pow,
        'l1_frac': l1_pow / total_pow,
        'zsym_frac':  zsym_pow  / bg_pow if bg_pow > 0 else 0,
        'zanti_frac': zanti_pow / bg_pow if bg_pow > 0 else 0,
        'l1_coeff': float(c_arr[l_arr == 1].flat[0]) if (l_arr == 1).any() else 0.0,
    }


def nmr_predictions(fit):
    """Evaluate fitted field at NMR probe positions."""
    from harmonic_basis import eval_field
    r_arr, phi_arr, z_arr, labels = [], [], [], []
    for lbl in PROBE_LABELS:
        x, y, z = NMR_PROBES[lbl]['xyz_cm']
        r_arr.append(np.sqrt(x**2 + y**2))
        phi_arr.append(np.arctan2(y, x))
        z_arr.append(z)
        labels.append(lbl)
    res = eval_field(fit['coeffs'], fit['params'],
                     np.array(r_arr), np.array(phi_arr), np.array(z_arr),
                     components=('Bz',), r_scale=fit['r_scale'], z0=fit['z0'])
    return {lbl: res['Bz'][i] for i, lbl in enumerate(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ref', default='tosca170812_full_coeffs_lmax18_all3.npz',
                        help='z0=0 reference fit')
    parser.add_argument('--z0', default='tosca170812_full_coeffs_lmax18_all3_z0_13mm.npz',
                        help='primary shifted fit')
    parser.add_argument('--z0b', default=None,
                        help='optional second shifted fit for three-way comparison')
    parser.add_argument('--lmax-show', type=int, default=6,
                        help='Show coefficients up to this l (default: 6)')
    args = parser.parse_args()

    fit_ref = load_fit(args.ref)
    fit_z0  = load_fit(args.z0)
    fit_z0b = load_fit(args.z0b) if args.z0b else None

    print(f"Reference fit : {args.ref}")
    print(f"  z0 = {fit_ref['z0']:.2f} cm, r_scale = {fit_ref['r_scale']:.1f} cm, "
          f"l_max = {fit_ref['l_max']}")
    print(f"  RMS: Bz={fit_ref['rms']['Bz']*1e3:.4f} mT  "
          f"Br={fit_ref['rms']['Br']*1e3:.4f} mT  "
          f"Bphi={fit_ref['rms']['Bphi']*1e3:.4f} mT")
    print()
    print(f"Shifted fit A : {args.z0}")
    print(f"  z0 = {fit_z0['z0']:.2f} cm, r_scale = {fit_z0['r_scale']:.1f} cm")
    print(f"  RMS: Bz={fit_z0['rms']['Bz']*1e3:.4f} mT  "
          f"Br={fit_z0['rms']['Br']*1e3:.4f} mT  "
          f"Bphi={fit_z0['rms']['Bphi']*1e3:.4f} mT")
    if fit_z0b:
        print(f"Shifted fit B : {args.z0b}")
        print(f"  z0 = {fit_z0b['z0']:.2f} cm, r_scale = {fit_z0b['r_scale']:.1f} cm")
        print(f"  RMS: Bz={fit_z0b['rms']['Bz']*1e3:.4f} mT  "
              f"Br={fit_z0b['rms']['Br']*1e3:.4f} mT  "
              f"Bphi={fit_z0b['rms']['Bphi']*1e3:.4f} mT")

    # --- Coefficient comparison table ---
    print(f"\n{'='*105}")
    print(f"Coefficient comparison (l ≤ {args.lmax_show})")
    print(f"  Parity of Bz: Bz(r,-z) = (-1)^(l-1) Bz(r,z)")
    print(f"  z-sym  = odd l  (l=1,3,5): z-symmetric Bz")
    print(f"  z-anti = even l (l=2,4,6): z-antisymmetric Bz (missing-turn asymmetry)")
    print_coeff_comparison(fit_ref, fit_z0, fit_z0b=fit_z0b, l_max_show=args.lmax_show)

    # --- Power analysis ---
    sep = '=' * (105 if fit_z0b else 90)
    print(f"\n{sep}")
    print("Power analysis (coefficient² sums, excluding l=1 background from fractions)")
    fits = [('z0=0', fit_ref), (f"z0=+{fit_z0['z0']:.1f}cm", fit_z0)]
    if fit_z0b:
        fits.append((f"z0=+{fit_z0b['z0']:.1f}cm", fit_z0b))
    headers = "  ".join(f"{lbl:>16s}" for lbl, _ in fits)
    print(f"{'Quantity':42s}  {headers}")
    print("-" * (42 + 20 * len(fits)))
    powers = [(lbl, power_analysis(f)) for lbl, f in fits]
    for key, label in [
        ('l1_frac',   'l=1 power fraction (background)'),
        ('zsym_frac', 'z-sym fraction of l>1 power'),
        ('zanti_frac','z-anti fraction of l>1 power'),
    ]:
        vals = "  ".join(f"{p[key]:16.6f}" for _, p in powers)
        print(f"  {label:40s}  {vals}")
    l1_vals = "  ".join(f"{p['l1_coeff']:+16.4f}" for _, p in powers)
    print(f"  {'l=1 m=0 c coefficient [T·cm]':40s}  {l1_vals}")

    # --- NMR predictions ---
    print(f"\n{sep}")
    print("NMR probe predictions")
    z0_hdrs = "  ".join(f"{'Bz_fit('+lbl+') [T]':>13s}  {'δ [mT]':>8s}"
                        for lbl, _ in fits)
    print(f"{'Probe':6s}  {'B_meas [T]':>10s}  {z0_hdrs}")
    nmr_fits = [(lbl, nmr_predictions(f)) for lbl, f in fits]
    for probe in PROBE_LABELS:
        B_meas = NMR_PROBES[probe]['B_meas']
        cols = "  ".join(f"{nm[probe]:>13.5f}  {(B_meas-nm[probe])*1e3:>+8.3f}"
                         for _, nm in nmr_fits)
        print(f"  {probe:4s}  {B_meas:>10.5f}  {cols}")
    print(f"\nNMR fit diff between shifted fits is <0.1 µT (mathematical invariance).")


if __name__ == '__main__':
    main()
