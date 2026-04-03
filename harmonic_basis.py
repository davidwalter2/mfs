"""
Maroussov harmonic polynomial basis for the CMS magnetic field scalar potential.

The scalar potential satisfying Laplace's equation in the tracker volume is expanded as:

  Phi(r, phi, z) = sum_{l=0}^{l_max} sum_{m=0}^{l}
                     [ A_{l,m} cos(m*phi) + B_{l,m} sin(m*phi) ] * R^l * P_l^m(cos(theta))

where R = sqrt(r^2 + z^2), cos(theta) = z/R, and P_l^m are the (non-normalized) associated
Legendre polynomials as returned by scipy.special.lpmv.

The field components are:

  Bz  = dPhi/dz  = sum (l+m) R^{l-1} P_{l-1}^m(cos(theta)) * [A cos(m*phi) + B sin(m*phi)]
  Br  = dPhi/dr  = sum R^{l-1} [l P_l^m - cos(theta)(l+m) P_{l-1}^m] / sin(theta)
                     * [A cos(m*phi) + B sin(m*phi)]
  Bphi = (1/r) dPhi/dphi = sum (m/r) R^l P_l^m(cos(theta))
                             * [-A sin(m*phi) + B cos(m*phi)]

The Bz formula is exact and singularity-free.
Br has a 1/sin(theta) factor which is handled carefully at r=0.
Bphi has 1/r, also zero at r=0 by L'Hopital (P_l^m ~ sin^m(theta) ~ r^m near r=0).

Reference: Maroussov (2008), PhD thesis, Purdue University.
See also CLAUDE.md "Maroussov Scalar Potential Approach".
"""

import numpy as np
from scipy.special import lpmv


def param_list(l_max, l_max_phi=None, n_max_sum=None, m_max_per_l=None):
    """Return list of (l, m, cs) tuples for all basis modes.

    cs = 'c' for cosine (A coefficient), 's' for sine (B coefficient, m>0 only).
    l=0 (Phi = constant) is omitted: it contributes zero to all field components
    (Bz = Br = Bphi = 0) and is a pure gauge degree of freedom that makes the
    design matrix rank-deficient.

    Four truncation schemes (checked in order; earlier ones take priority):
    ───────────────────────────────────────────────────────────────────────
    Custom per-l envelope  (m_max_per_l dict set)
        m_max_per_l maps l → max_m for that l; l absent from the dict uses
        the default max_m = min(l, l_max_phi if l_max_phi else l_max).
        Set m_max_per_l[l] = -1 to skip that l entirely.
        Example: {17: -1, 16: 0, 18: 0, 6: 2, ..., 15: 1}

    Diamond  (n_max_sum set, l_max_phi ignored)
        Include (l, m) if l + m ≤ n_max_sum (and m ≤ l, l ≥ 1).
        l_max is automatically set to n_max_sum.

    Asymmetric  (l_max_phi set, n_max_sum=None)
        m=0 modes up to l=l_max; m>0 modes up to l=l_max_phi.
        Total: l_max + l_max_phi*(l_max_phi+1) parameters.

    Standard triangular  (all None)
        Include (l, m) for l=1..l_max, m=0..l.
        Total: (l_max+1)^2 − 1 parameters.
    """
    if m_max_per_l is not None:
        default_mmax = l_max_phi if l_max_phi is not None else l_max
        params = []
        for l in range(1, l_max + 1):
            mmax = m_max_per_l.get(l, min(l, default_mmax))
            if mmax < 0:
                continue   # skip this l entirely
            mmax = min(mmax, l)
            params.append((l, 0, 'c'))
            for m in range(1, mmax + 1):
                params.append((l, m, 'c'))
                params.append((l, m, 's'))
        return params

    if n_max_sum is not None:
        params = []
        for l in range(1, n_max_sum + 1):
            for m in range(0, l + 1):
                if l + m <= n_max_sum:
                    params.append((l, m, 'c'))
                    if m > 0:
                        params.append((l, m, 's'))
        return params

    if l_max_phi is None:
        l_max_phi = l_max
    params = []
    for l in range(1, l_max + 1):
        params.append((l, 0, 'c'))
    for l in range(1, l_max_phi + 1):
        for m in range(1, l + 1):
            params.append((l, m, 'c'))
            params.append((l, m, 's'))
    return params


def _plm(l, m, cos_theta):
    """Associated Legendre polynomial P_l^m(cos_theta), vectorized.
    Uses scipy convention (no Condon-Shortley phase).
    Returns zero for l < 0.
    """
    if l < 0:
        return np.zeros_like(cos_theta)
    return lpmv(m, l, cos_theta)


def bz_basis(l, m, cs, r, phi, z, r_scale=1.0, z0=0.0):
    """Bz component of the (l, m, cs) basis function.

    Bz = (l+m) (R/r_scale)^{l-1} / r_scale * P_{l-1}^m(cos(theta))
         * [cos(m*phi) if cs='c', sin(m*phi) if cs='s']

    For l=0: Bz = 0 (gauge mode).

    Parameters
    ----------
    l, m : int  degree and order
    cs   : 'c' or 's'  cosine or sine phi-dependence
    r, phi, z : array_like  cylindrical coordinates (r in same units as z)
    r_scale : float  normalisation radius; set to max(R) of fitting region for stability
    z0 : float  axial offset of expansion origin (same units as z); use the
                magnetic centre offset to centre the expansion on the field axis.
                z' = z - z0 is used internally; default 0 (origin at z=0).

    Returns
    -------
    ndarray, same shape as r/phi/z inputs
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float) - z0  # shift to expansion origin

    if l == 0:
        return np.zeros_like(r)

    R = np.sqrt(r**2 + z**2)
    Rn = R / r_scale  # normalised spherical radius, ≤ 1 within fitting region
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = np.where(R > 0, z / R, np.sign(z))  # ±1 on axis; 0 at exact origin

    # (R/r_scale)^(l-1) / r_scale: for l=1 this is 1/r_scale everywhere
    if l == 1:
        R_pow = np.full_like(R, 1.0 / r_scale)
    else:
        R_pow = np.where(R > 0, Rn**(l - 1) / r_scale, 0.0)

    prefactor = (l + m) * R_pow
    plm_val = _plm(l - 1, m, cos_theta)

    phi_factor = np.cos(m * phi) if cs == 'c' else np.sin(m * phi)

    return prefactor * plm_val * phi_factor


def br_basis(l, m, cs, r, phi, z, r_scale=1.0, z0=0.0):
    """Br component of the (l, m, cs) basis function.
    For l=0: Br = 0 (gauge mode).

    Br = (R/r_scale)^{l-1} / r_scale
         * [l P_l^m(cos_theta) - cos_theta*(l+m)*P_{l-1}^m(cos_theta)] / sin_theta
         * phi_factor

    At sin_theta=0 (r=0): Br=0 by L'Hopital (numerator also vanishes).

    Parameters: same as bz_basis (z0 shifts expansion origin along z).
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float) - z0

    if l == 0:
        return np.zeros_like(r)

    R = np.sqrt(r**2 + z**2)
    Rn = R / r_scale
    with np.errstate(invalid='ignore', divide='ignore'):
        sin_theta = np.where(R > 0, r / R, 0.0)
        cos_theta = np.where(R > 0, z / R, np.sign(z))

    plm_l = _plm(l, m, cos_theta)
    plm_lm1 = _plm(l - 1, m, cos_theta)

    numerator = l * plm_l - cos_theta * (l + m) * plm_lm1
    # Safely divide; at r=0 both numerator and sin_theta vanish → 0
    safe = sin_theta > 1e-12
    if l == 1:
        R_pow = np.full_like(R, 1.0 / r_scale)
    else:
        R_pow = np.where(R > 0, Rn**(l - 1) / r_scale, 0.0)
    br_val = np.where(safe,
                      R_pow * numerator / np.where(safe, sin_theta, 1.0),
                      0.0)

    phi_factor = np.cos(m * phi) if cs == 'c' else np.sin(m * phi)

    return br_val * phi_factor


def bphi_basis(l, m, cs, r, phi, z, r_scale=1.0, z0=0.0):
    """Bphi component of the (l, m, cs) basis function.

    Bphi = (1/r) dPhi/dphi = -(m/r) (R/r_scale)^l P_l^m(cos_theta) * sin(m*phi) [cs='c']
                           = +(m/r) (R/r_scale)^l P_l^m(cos_theta) * cos(m*phi)  [cs='s']

    At r=0: Bphi=0 (P_l^m ~ r^m, so P_l^m/r ~ r^{m-1} → 0 for m≥1, and m=0 → sin=0).

    Parameters: same as bz_basis (z0 shifts expansion origin along z).
    """
    if m == 0:
        return np.zeros_like(np.asarray(r, dtype=float))

    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float) - z0

    R = np.sqrt(r**2 + z**2)
    Rn = R / r_scale
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = np.where(R > 0, z / R, np.sign(z))

    plm_val = _plm(l, m, cos_theta)
    R_l = np.where(R > 0, Rn**l, 0.0)

    # m/r * (R/r_scale)^l * P_l^m; near r=0: P_l^m ~ sin^m(theta) ~ (r/R)^m → 0 for m≥1
    safe_r = r > 1e-12
    inv_r = np.where(safe_r, 1.0 / np.where(safe_r, r, 1.0), 0.0)

    if cs == 'c':
        phi_factor = -np.sin(m * phi)
    else:
        phi_factor = +np.cos(m * phi)

    return m * inv_r * R_l * plm_val * phi_factor


def build_design_matrix(r, phi, z, l_max, components=('Bz',), r_scale=None, z0=0.0,
                        l_max_phi=None, n_max_sum=None, m_max_per_l=None):
    """Build the design matrix for the harmonic basis fit.

    Each row corresponds to one data point (one field component at one location).
    Each column corresponds to one basis coefficient.

    Parameters
    ----------
    r, phi, z  : 1-D array_like  cylindrical coordinates of grid points
    l_max      : int             maximum degree
    components : tuple of str   which field components to include ('Bz', 'Br', 'Bphi')
    r_scale    : float or None   normalisation radius for numerical stability.
                                 If None, uses max(sqrt(r^2+(z-z0)^2)) of the input points.
    z0         : float           axial offset of expansion origin (same units as z);
                                 use the magnetic centre shift to reduce odd-l content.

    Returns
    -------
    A      : ndarray, shape (n_data, n_params)
    params : list of (l, m, cs) tuples, length n_params
    r_scale: float, the normalisation radius used (needed for eval_field)
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float)

    if r_scale is None:
        zp = z - z0
        r_scale = np.max(np.sqrt(r**2 + zp**2))
        if r_scale == 0:
            r_scale = 1.0

    params = param_list(l_max, l_max_phi=l_max_phi, n_max_sum=n_max_sum,
                        m_max_per_l=m_max_per_l)
    n_pts = len(r)
    n_comp = len(components)
    n_params = len(params)

    A = np.zeros((n_pts * n_comp, n_params))

    comp_funcs = {'Bz': bz_basis, 'Br': br_basis, 'Bphi': bphi_basis}

    for ic, comp in enumerate(components):
        func = comp_funcs[comp]
        row_start = ic * n_pts
        for ip, (l, m, cs) in enumerate(params):
            A[row_start:row_start + n_pts, ip] = func(l, m, cs, r, phi, z,
                                                       r_scale=r_scale, z0=z0)

    return A, params, r_scale


def eval_field(coeffs, params, r, phi, z, components=('Bz',), r_scale=1.0, z0=0.0):
    """Evaluate the fitted field at arbitrary points.

    Parameters
    ----------
    coeffs  : 1-D array  fitted coefficients (length = len(params))
    params  : list of (l, m, cs) tuples
    r, phi, z : array_like  evaluation points
    components : tuple of str  which components to return
    r_scale : float  same normalisation radius used during fitting
    z0      : float  same axial offset used during fitting

    Returns
    -------
    dict mapping component name to ndarray of values
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float)

    comp_funcs = {'Bz': bz_basis, 'Br': br_basis, 'Bphi': bphi_basis}
    result = {}

    for comp in components:
        func = comp_funcs[comp]
        val = np.zeros_like(r)
        for c, (l, m, cs) in zip(coeffs, params):
            if c != 0:
                val += c * func(l, m, cs, r, phi, z, r_scale=r_scale, z0=z0)
        result[comp] = val

    return result
