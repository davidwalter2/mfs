"""
Cylindrical Fourier-Bessel harmonic basis for the CMS magnetic field scalar potential.

The scalar potential satisfying Laplace's equation inside the solenoid bore is expanded as:

  Phi(r, phi, z) = c_0 * z                             [uniform Bz, n=0 m=0]
                 + sum_{n>=1, m>=0}
                     I_m(k_n r) [A cos(m phi) + B sin(m phi)]
                                [C sin(k_n z) + D cos(k_n z)]

where k_n = n*pi/L for a chosen half-period L, and I_m is the modified Bessel function
of the first kind (regular at r=0).

The field components are:

  Bz   = dPhi/dz  = c_0                                     [n=0]
                  + k_n I_m(k_n r) [cos/sin mφ] [+cos(k_n z) or -sin(k_n z)]

  Br   = dPhi/dr  = k_n I_m'(k_n r) [cos/sin mφ] [sin(k_n z) or cos(k_n z)]
         where I_m'(x) = (I_{m-1}(x) + I_{m+1}(x)) / 2

  Bphi = (1/r) dPhi/dphi = (I_m(k_n r)/r) m [-sin/+cos mφ] [sin/cos k_n z]

Z-parity classification (for the solenoid, Bz is nearly z-symmetric):
  cz='sym'  → Phi ~ I_m sin(k_n z) → Bz ~ k_n I_m cos(k_n z)  [z-symmetric Bz]
  cz='anti' → Phi ~ I_m cos(k_n z) → Bz ~ -k_n I_m sin(k_n z) [z-antisymmetric Bz]

Mode tuple: (n, m, cphi, cz)
  n    : wavenumber index 0,1,...,n_max
  m    : azimuthal order 0,1,...,m_max
  cphi : 'c' = cos(m phi), 's' = sin(m phi)  [only 'c' used for m=0]
  cz   : 'sym' or 'anti'

Parameter label format: "n{n}_m{m}_{cphi}_{cz}"  e.g. "n1_m0_c_sym"
Special: n=0 uniform mode labelled "n0_m0_c_sym"

Note on conditioning:
  I_m(k_n r) grows exponentially for k_n r >> 1.  Column normalisation in fit_field.py
  handles this: each column is divided by its L2 norm so all columns have unit norm.
  This means the fitted coefficients are in "normalised" units and must be divided by
  col_norms to recover physical (Tesla) coefficients.

Reference: Jackson, Classical Electrodynamics, Ch.3; Maroussov (2008) Section 6.
"""

import numpy as np
from scipy.special import iv   # modified Bessel I_v(x)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _Im(m, x):
    """I_m(x) via scipy.special.iv(m, x)."""
    return iv(m, x)


def _dImdr(m, k, r):
    """d/dr I_m(k r) = k * I_m'(kr), using I_m'(x) = (I_{m-1}(x)+I_{m+1}(x))/2.
    At r=0 returns the limiting value (finite for m<=1)."""
    x = np.asarray(k * r, dtype=float)
    # I_{-1}(x) = I_1(x)
    Im1 = iv(max(m - 1, 0) if np.isscalar(m) else m - 1, x)
    if np.isscalar(m) and m == 0:
        Im1 = iv(1, x)   # I_{-1} = I_1
    elif not np.isscalar(m):
        Im1 = iv(np.abs(m - 1), x)
    Ip1 = iv(m + 1, x)
    return k * 0.5 * (Im1 + Ip1)


def _phi_factor(m, cphi, phi):
    if m == 0:
        return np.ones_like(phi)
    return np.cos(m * phi) if cphi == 'c' else np.sin(m * phi)


def _dphi_factor(m, cphi, phi):
    """(1/r) dPhi/dphi azimuthal factor (without the 1/r)."""
    if m == 0:
        return np.zeros_like(phi)
    return -m * np.sin(m * phi) if cphi == 'c' else m * np.cos(m * phi)


# ---------------------------------------------------------------------------
# mode enumeration
# ---------------------------------------------------------------------------

def param_list_cyl(n_max, m_max):
    """Return list of (n, m, cphi, cz) mode tuples.

    n=0: only the uniform Bz mode (n=0, m=0, 'c', 'sym').
    n>=1: all combinations of m in [0,m_max], cphi in {'c'} for m=0
          or {'c','s'} for m>=1, and cz in {'sym','anti'}.

    Total modes: 1  +  2*n_max*(1 + 2*m_max)
    """
    modes = []
    modes.append((0, 0, 'c', 'sym'))   # uniform Bz
    for n in range(1, n_max + 1):
        for m in range(0, m_max + 1):
            cphi_vals = ['c'] if m == 0 else ['c', 's']
            for cphi in cphi_vals:
                for cz in ['sym', 'anti']:
                    modes.append((n, m, cphi, cz))
    return modes


def mode_label(n, m, cphi, cz):
    return f"n{n}_m{m}_{cphi}_{cz}"


def parse_mode_label(s):
    """Parse "n{n}_m{m}_{cphi}_{cz}" → (n, m, cphi, cz)."""
    parts = s.split('_')
    n  = int(parts[0][1:])
    m  = int(parts[1][1:])
    cphi = parts[2]
    cz   = parts[3]
    return n, m, cphi, cz


# ---------------------------------------------------------------------------
# basis functions
# ---------------------------------------------------------------------------

def bz_cyl_basis(n, m, cphi, cz, r, phi, z, L):
    """Bz = dPhi/dz for cylindrical harmonic mode (n, m, cphi, cz).

    Parameters
    ----------
    n, m, cphi, cz : mode quantum numbers
    r, phi, z : arrays [cm]
    L : half-period [cm]

    Returns array of shape r.shape.
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    phi_fac = _phi_factor(m, cphi, phi)

    if n == 0:
        # Phi = z  →  Bz = 1 (uniform)
        return phi_fac * np.ones_like(r)   # only non-zero for m=0 (phi_fac=1)

    k = n * np.pi / L
    Im = iv(m, k * r)

    if cz == 'sym':
        # Phi ~ I_m(kr) sin(kz) → Bz = k I_m cos(kz)
        return k * Im * phi_fac * np.cos(k * z)
    else:
        # Phi ~ I_m(kr) cos(kz) → Bz = -k I_m sin(kz)
        return -k * Im * phi_fac * np.sin(k * z)


def br_cyl_basis(n, m, cphi, cz, r, phi, z, L):
    """Br = dPhi/dr for cylindrical harmonic mode."""
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    phi_fac = _phi_factor(m, cphi, phi)

    if n == 0:
        # Phi = z  →  Br = 0
        return np.zeros_like(r)

    k = n * np.pi / L
    dIdr = _dImdr(m, k, r)

    if cz == 'sym':
        # Phi ~ I_m(kr) sin(kz) → Br = I_m'(kr) * k * ... wait: d/dr[I_m(kr)] already = k*I_m'
        # Actually dIdr already includes the k factor from _dImdr.
        return dIdr * phi_fac * np.sin(k * z)
    else:
        # Phi ~ I_m(kr) cos(kz) → Br = dI/dr cos(kz)
        return dIdr * phi_fac * np.cos(k * z)


def bphi_cyl_basis(n, m, cphi, cz, r, phi, z, L):
    """Bphi = (1/r) dPhi/dphi for cylindrical harmonic mode."""
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    if m == 0 or n == 0:
        return np.zeros_like(r)

    k = n * np.pi / L
    Im = iv(m, k * r)
    dphi_fac = _dphi_factor(m, cphi, phi)

    if cz == 'sym':
        z_fac = np.sin(k * z)
    else:
        z_fac = np.cos(k * z)

    # Safe 1/r: I_m(kr)/r → (kr)^m / (2^m m! r) = k^m r^{m-1}/(2^m m!) as r→0, finite for m>=1
    r_safe = np.where(r > 0, r, np.ones_like(r))
    result = np.where(r > 0,
                      (Im / r_safe) * dphi_fac * z_fac,
                      np.zeros_like(r))
    return result


# ---------------------------------------------------------------------------
# design matrix and field evaluation
# ---------------------------------------------------------------------------

def build_design_matrix_cyl(r, phi, z, n_max, m_max, L,
                             components=('Bz', 'Br', 'Bphi')):
    """Build the design matrix A for the cylindrical harmonic fit.

    Shape: (N_points * N_components, N_modes)
    where N_modes = 1 + 2*n_max*(1 + 2*m_max).

    Parameters
    ----------
    r, phi, z   : coordinate arrays [cm], length N
    n_max       : maximum wavenumber index
    m_max       : maximum azimuthal order
    L           : Fourier half-period [cm]
    components  : tuple of 'Bz', 'Br', 'Bphi'

    Returns
    -------
    A      : ndarray, shape (N*n_comp, N_modes)
    modes  : list of (n, m, cphi, cz) tuples
    """
    modes = param_list_cyl(n_max, m_max)
    N = len(r)
    n_modes = len(modes)
    n_comp = len(components)

    A = np.zeros((N * n_comp, n_modes))

    basis_fns = {
        'Bz':   bz_cyl_basis,
        'Br':   br_cyl_basis,
        'Bphi': bphi_cyl_basis,
    }

    for j, (n, m, cphi, cz) in enumerate(modes):
        col_blocks = []
        for comp in components:
            col = basis_fns[comp](n, m, cphi, cz, r, phi, z, L)
            col_blocks.append(col)
        A[:, j] = np.concatenate(col_blocks)

    return A, modes


def eval_field_cyl(coeffs, modes, r, phi, z, L,
                   components=('Bz', 'Br', 'Bphi')):
    """Evaluate the cylindrical harmonic field at arbitrary points.

    Parameters
    ----------
    coeffs  : 1D array of fitted coefficients, length N_modes
    modes   : list of (n, m, cphi, cz) tuples (from param_list_cyl or loaded from npz)
    r, phi, z : coordinate arrays [cm]
    L       : Fourier half-period [cm]
    components : which field components to return

    Returns dict {comp: array}.
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    result = {comp: np.zeros_like(r) for comp in components}

    basis_fns = {
        'Bz':   bz_cyl_basis,
        'Br':   br_cyl_basis,
        'Bphi': bphi_cyl_basis,
    }

    for c, (n, m, cphi, cz) in zip(coeffs, modes):
        if c == 0.0:
            continue
        for comp in components:
            result[comp] += c * basis_fns[comp](n, m, cphi, cz, r, phi, z, L)

    return result
