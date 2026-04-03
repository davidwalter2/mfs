"""
cylindrical_bessel_basis.py — J_m Bessel + hyperbolic (cosh/sinh) z basis.

Unlike the Fourier-Bessel basis (modified Bessel I_m times oscillatory sin/cos in z),
this basis uses the *other* separation-of-variables choice for Laplace's equation:

  Phi(r, phi, z) = c_0 * z                                  [uniform Bz, n=0]
                 + sum_{n>=1, m>=0}
                     J_m(alpha_{m,n} r / R)
                     * [cosh(alpha_{m,n} z / R)  or  sinh(alpha_{m,n} z / R)]
                     * [A cos(m phi)  +  B sin(m phi)]

where:
  J_m           = regular Bessel function of the first kind
  alpha_{m,n}   = n-th positive zero of J_m   (from scipy.special.jn_zeros)
  R             = radial scale [cm]   (set to r_max of the fitting domain)

Motivation
----------
The CMS solenoid field is smooth and monotonic in z — it has no oscillations.
The Fourier-Bessel basis (I_m sin(k_n z)) requires many z-harmonics to represent
a monotonic function, giving slow convergence (~200× worse than spherical harmonics
at equal parameter count).

Replacing sin/cos → cosh/sinh in z gives basis functions that are monotonic in z,
matching the actual field shape.  The radial part switches from I_m (grows
exponentially) to J_m (oscillatory but bounded), discretized at the zeros of J_m so
that {J_m(alpha_{m,n} r/R)} form a complete orthogonal set on [0, R].

Field components
----------------
For a mode (n>=1, m, cphi, cz) with k = alpha_{m,n} / R:

  Bz   = dPhi/dz = k  J_m(kr)   sinh(kz)  phi_fac   [cz='sym' ]
                 = k  J_m(kr)   cosh(kz)  phi_fac   [cz='anti']

  Br   = dPhi/dr = k  J_m'(kr)  cosh(kz)  phi_fac   [cz='sym' ]
                 = k  J_m'(kr)  sinh(kz)  phi_fac   [cz='anti']
         where J_m'(x) = [J_{m-1}(x) - J_{m+1}(x)] / 2

  Bphi = (1/r) dPhi/dphi = (m/r) J_m(kr)  cosh(kz)  dphi_fac  [cz='sym' ]
                          = (m/r) J_m(kr)  sinh(kz)  dphi_fac  [cz='anti']
         where dphi_fac = -sin(m phi) for cphi='c',  +cos(m phi) for cphi='s'
         Bphi = 0 for m=0.

For the n=0 uniform mode: Bz = 1, Br = 0, Bphi = 0  (Phi = z).

Numerical note
--------------
cosh(k z_max) grows exponentially for large k.  For k = alpha_{0,5}/R ~ 15/290
and z_max=300 cm, cosh(kz_max) ~ 1e6.  Column normalisation in fit_field.py
(dividing each design-matrix column by its L2 norm) handles this automatically;
the condition number of the normalised matrix is what determines fit quality.

Mode label format: "n{n}_m{m}_{cphi}_{cz}"  (same as cylindrical_basis.py)
"""

import numpy as np
from scipy.special import jv, jn_zeros


# ── helpers ──────────────────────────────────────────────────────────────────

def _Jm(m, x):
    return jv(m, x)

def _dJm(m, x):
    """Derivative J_m'(x) = [J_{m-1}(x) - J_{m+1}(x)] / 2."""
    return 0.5 * (jv(m - 1, x) - jv(m + 1, x))

def _phi_fac(m, cphi, phi):
    if m == 0:
        return np.ones_like(phi)
    return np.cos(m * phi) if cphi == 'c' else np.sin(m * phi)

def _dphi_fac(m, cphi, phi):
    """Angular factor for Bphi: d/dphi[phi_fac] without the m prefactor."""
    if m == 0:
        return np.zeros_like(phi)
    return -np.sin(m * phi) if cphi == 'c' else np.cos(m * phi)


# ── parameter list ────────────────────────────────────────────────────────────

def param_list_cjb(n_max, m_max):
    """Return list of (n, m, cphi, cz) mode tuples.

    n=0 : uniform Bz mode  (Phi = z)
    n>=1: all (m, cphi, cz) combinations; cphi in {'c'} for m=0, {'c','s'} for m>0

    Total modes: 1 + 2*n_max*(1 + 2*m_max)  — same count as Fourier-Bessel.
    """
    modes = [(0, 0, 'c', 'sym')]   # uniform Bz
    for n in range(1, n_max + 1):
        for m in range(0, m_max + 1):
            cphi_list = ['c'] if m == 0 else ['c', 's']
            for cphi in cphi_list:
                for cz in ['sym', 'anti']:
                    modes.append((n, m, cphi, cz))
    return modes


def mode_label(n, m, cphi, cz):
    return f"n{n}_m{m}_{cphi}_{cz}"

def parse_mode_label(s):
    parts = s.split('_')
    return int(parts[0][1:]), int(parts[1][1:]), parts[2], parts[3]


# ── basis functions ───────────────────────────────────────────────────────────

def bz_cjb_basis(n, m, cphi, cz, r, phi, z, R, zeros_cache):
    """Bz = dPhi/dz for mode (n, m, cphi, cz)."""
    pf = _phi_fac(m, cphi, phi)
    if n == 0:
        return pf   # m=0 → pf=1 everywhere → uniform Bz=1

    k  = zeros_cache[m][n - 1] / R
    Jm = _Jm(m, k * r)
    if cz == 'sym':
        return k * Jm * np.sinh(k * z) * pf
    else:
        return k * Jm * np.cosh(k * z) * pf


def br_cjb_basis(n, m, cphi, cz, r, phi, z, R, zeros_cache):
    """Br = dPhi/dr for mode (n, m, cphi, cz)."""
    if n == 0:
        return np.zeros_like(r)

    pf  = _phi_fac(m, cphi, phi)
    k   = zeros_cache[m][n - 1] / R
    dJm = _dJm(m, k * r)
    if cz == 'sym':
        return k * dJm * np.cosh(k * z) * pf
    else:
        return k * dJm * np.sinh(k * z) * pf


def bphi_cjb_basis(n, m, cphi, cz, r, phi, z, R, zeros_cache):
    """Bphi = (1/r) dPhi/dphi for mode (n, m, cphi, cz)."""
    if m == 0 or n == 0:
        return np.zeros_like(r)

    dpf = _dphi_fac(m, cphi, phi)
    k   = zeros_cache[m][n - 1] / R
    Jm  = _Jm(m, k * r)
    safe = r > 1e-12
    inv_r = np.where(safe, 1.0 / np.where(safe, r, 1.0), 0.0)

    if cz == 'sym':
        return m * inv_r * Jm * np.cosh(k * z) * dpf
    else:
        return m * inv_r * Jm * np.sinh(k * z) * dpf


# ── design matrix ─────────────────────────────────────────────────────────────

def build_design_matrix_cjb(r, phi, z, n_max, m_max, R,
                             components=('Bz', 'Br', 'Bphi')):
    """Build design matrix for the J_m + cosh/sinh basis.

    Parameters
    ----------
    r, phi, z   : coordinate arrays [cm], length N
    n_max       : maximum Bessel zero index (1st … n_max-th zero of each J_m)
    m_max       : maximum azimuthal order
    R           : radial scale [cm] — Bessel zeros are alpha_{m,n}/R
    components  : tuple of 'Bz', 'Br', 'Bphi'

    Returns
    -------
    A     : ndarray, shape (N * n_comp, N_modes)
    modes : list of (n, m, cphi, cz) tuples
    """
    modes = param_list_cjb(n_max, m_max)
    N      = len(r)
    n_comp = len(components)
    A      = np.zeros((N * n_comp, len(modes)))

    # Pre-compute all needed Bessel zeros once
    zeros_cache = {m: jn_zeros(m, n_max) for m in range(0, m_max + 1)}

    basis_fns = {
        'Bz':   bz_cjb_basis,
        'Br':   br_cjb_basis,
        'Bphi': bphi_cjb_basis,
    }

    for j, (n, m, cphi, cz) in enumerate(modes):
        col_blocks = [basis_fns[comp](n, m, cphi, cz, r, phi, z, R, zeros_cache)
                      for comp in components]
        A[:, j] = np.concatenate(col_blocks)

    return A, modes


# ── field evaluation ──────────────────────────────────────────────────────────

def eval_field_cjb(coeffs, modes, r, phi, z, R, components=('Bz', 'Br', 'Bphi')):
    """Evaluate the fitted field at arbitrary points.

    Parameters
    ----------
    coeffs    : 1-D array, length = len(modes)
    modes     : list of (n, m, cphi, cz) tuples
    r, phi, z : evaluation coordinates [cm]
    R         : radial scale used during fitting [cm]
    components: which components to return

    Returns
    -------
    dict mapping component name → ndarray
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    n_max = max(n for n, *_ in modes)
    m_max = max(m for _, m, *_ in modes)
    zeros_cache = {m: jn_zeros(m, max(n_max, 1)) for m in range(0, m_max + 1)}

    basis_fns = {'Bz': bz_cjb_basis, 'Br': br_cjb_basis, 'Bphi': bphi_cjb_basis}
    result = {comp: np.zeros_like(r) for comp in components}

    for c, (n, m, cphi, cz) in zip(coeffs, modes):
        if c == 0.0:
            continue
        for comp in components:
            result[comp] += c * basis_fns[comp](n, m, cphi, cz, r, phi, z, R, zeros_cache)

    return result
