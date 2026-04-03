"""
Inspect the radial profile of Bz near the coil inner wall to check for TOSCA
interpolation artifacts before extending the harmonic fit to r~290 cm.

Prints Bz, Br, |dBz/dr| at z=0 (and optionally other z slices) vs r,
comparing the existing 'full' grid (rMax=280 cm) against the 'extended'
grid (rMax=295 cm) to check for discontinuities.

Usage:
  python inspect_radial.py
  python inspect_radial.py --extended field_tosca170812_extended.txt
  python inspect_radial.py --full field_tosca170812_full.txt --extended field_tosca170812_extended.txt
"""

import numpy as np
import argparse


def load_grid(path):
    data = np.loadtxt(path, comments='#')
    r, phi, z = data[:,0], data[:,1], data[:,2]
    Bx, By, Bz = data[:,5], data[:,6], data[:,7]
    Br   =  Bx * np.cos(phi) + By * np.sin(phi)
    Bphi = -Bx * np.sin(phi) + By * np.cos(phi)
    return {'r': r, 'phi': phi, 'z': z, 'Bz': Bz, 'Br': Br, 'Bphi': Bphi,
            'Bx': Bx, 'By': By}


def radial_slice(grid, z_target=0.0, phi_target=0.0, dz=5.0, dphi=0.1):
    """Extract a radial slice near z=z_target, phi=phi_target."""
    mask = (np.abs(grid['z'] - z_target) < dz) & (np.abs(grid['phi'] - phi_target) < dphi)
    r    = grid['r'][mask]
    Bz   = grid['Bz'][mask]
    Br   = grid['Br'][mask]
    Bphi = grid['Bphi'][mask]
    idx  = np.argsort(r)
    return r[idx], Bz[idx], Br[idx], Bphi[idx]


def print_radial(r, Bz, Br, Bphi, label, r_lo=250.0):
    """Print field values for r >= r_lo, with numerical dBz/dr."""
    mask = r >= r_lo
    r, Bz, Br, Bphi = r[mask], Bz[mask], Br[mask], Bphi[mask]
    if len(r) < 2:
        print(f"  [no points at r >= {r_lo} cm in {label}]")
        return

    dBzdr = np.gradient(Bz, r)

    print(f"\n  {label}  (r ≥ {r_lo:.0f} cm, z≈0, phi≈0)")
    print(f"  {'r[cm]':>8s}  {'Bz[T]':>10s}  {'Br[T]':>10s}  {'Bphi[T]':>10s}  "
          f"{'dBz/dr [T/cm]':>14s}  {'|dBz/dr|×r_scale [T]':>22s}")
    r_scale = 280.0
    for i in range(len(r)):
        # Flag points where |dBz/dr| > 0.01 T/cm as potential artifacts
        flag = ' *** LARGE GRADIENT' if abs(dBzdr[i]) > 0.01 else ''
        print(f"  {r[i]:8.1f}  {Bz[i]:10.5f}  {Br[i]:10.5f}  {Bphi[i]:10.6f}  "
              f"{dBzdr[i]:14.5f}  {dBzdr[i]*r_scale:22.4f}{flag}")


def check_continuity(r, Bz, r_boundary=280.0, window=5.0):
    """Check for discontinuity near r_boundary by comparing left/right extrapolations."""
    left  = (r >= r_boundary - window) & (r < r_boundary)
    right = (r >= r_boundary) & (r <= r_boundary + window)
    if left.sum() < 2 or right.sum() < 2:
        return None, None, None
    # Linear extrapolation from each side to r_boundary
    p_l = np.polyfit(r[left],  Bz[left],  1)
    p_r = np.polyfit(r[right], Bz[right], 1)
    Bz_left_extrap  = np.polyval(p_l, r_boundary)
    Bz_right_extrap = np.polyval(p_r, r_boundary)
    jump = Bz_right_extrap - Bz_left_extrap
    return Bz_left_extrap, Bz_right_extrap, jump


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--full',     default='../MagneticField/Engine/test/field_tosca170812_full.txt',
                        help='Existing full grid (rMax=280 cm)')
    parser.add_argument('--extended', default='../MagneticField/Engine/test/field_tosca170812_extended.txt',
                        help='Extended grid (rMax=295 cm)')
    parser.add_argument('--r-lo', type=float, default=230.0,
                        help='Minimum r [cm] to show in radial table (default: 230)')
    args = parser.parse_args()

    import os

    grids = {}
    for name, path in [('full (rMax=280)', args.full), ('extended (rMax=295)', args.extended)]:
        if os.path.exists(path):
            grids[name] = load_grid(path)
            print(f"Loaded {name}: {path}  ({len(grids[name]['r'])} points, "
                  f"r up to {grids[name]['r'].max():.1f} cm)")
        else:
            print(f"[missing] {name}: {path}")

    if not grids:
        print("No grid files found.")
        return

    print(f"\n{'='*100}")
    print("Radial profile at z≈0, phi≈0  (checking for artifacts near coil inner wall ~290 cm)")
    print("  |dBz/dr| > 0.01 T/cm flagged as LARGE GRADIENT")

    for name, grid in grids.items():
        r, Bz, Br, Bphi = radial_slice(grid, z_target=0.0, phi_target=0.0, dz=5.0, dphi=0.1)
        print_radial(r, Bz, Br, Bphi, name, r_lo=args.r_lo)

    # Continuity check at r=280 cm using the extended grid
    if 'extended (rMax=295)' in grids:
        grid = grids['extended (rMax=295)']
        r, Bz, Br, Bphi = radial_slice(grid, z_target=0.0, phi_target=0.0, dz=5.0, dphi=0.1)
        Bz_l, Bz_r, jump = check_continuity(r, Bz, r_boundary=280.0, window=10.0)
        if jump is not None:
            print(f"\n{'='*100}")
            print(f"Continuity check at r=280 cm (grid boundary of 'full' fit):")
            print(f"  Linear extrapolation from r<280: Bz = {Bz_l:.5f} T")
            print(f"  Linear extrapolation from r>280: Bz = {Bz_r:.5f} T")
            print(f"  Jump at r=280 cm: {jump*1e3:+.3f} mT")
            if abs(jump) < 1e-4:
                print(f"  → Smooth (< 0.1 mT jump) — safe to include r=280-295 cm in fit")
            elif abs(jump) < 1e-3:
                print(f"  → Small but visible discontinuity (~{abs(jump)*1e3:.1f} mT) — use with caution")
            else:
                print(f"  → Large discontinuity ({abs(jump)*1e3:.1f} mT) — DO NOT include in harmonic fit")

    print(f"\n{'='*100}")
    print("NMR probe radii for reference:")
    print("  Probes A, E: r = 291.5 cm, z ≈ 0       (at coil boundary)")
    print("  Probes C, D: r =  65.1 cm, z ≈ ±283 cm (in tracker endcap region)")
    print("  Coil inner bore: r ≈ 290 cm")


if __name__ == '__main__':
    main()
