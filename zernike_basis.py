"""
zernike_basis.py — 3D Zernike polynomial scalar-potential basis for magnetic field fitting.

The scalar potential is expanded as:

  Phi(R, theta, phi) = sum_{n,l,m} c_{nlm} Z_{n,l,m}(R, theta, phi)

  Z_{n,l,m} = R_n^l(rho) * Y_l^m_real(theta, phi)

where rho = R/r_scale, R = sqrt(r_cyl^2 + z^2), and:

  R_n^l(rho) = rho^l * P_{(n-l)/2}^{(0, l+1/2)}(2*rho^2 - 1)

P_k^(alpha,beta) is the Jacobi polynomial.  The Z_{nlm} are orthogonal on the unit
ball with the standard volume measure dV = R^2 dR sin(theta) dtheta dphi.

Allowed modes: n >= 1, 0 <= l <= min(n, l_max), n-l even, 0 <= m <= l.
n=0 (gauge mode Phi=const -> B=0) is excluded.

Relation to the standard harmonic basis
----------------------------------------
For k = (n-l)/2 = 0  (i.e. n = l):  R_n^l(rho) = rho^l  (the regular solid harmonic).
The n=l modes are therefore identical to the standard Maroussov spherical harmonic modes.
Additional modes with k >= 1 provide extra radial depth without increasing the maximum
azimuthal order m, enabling a "pencil" truncation:

  Standard harmonic:  l <= l_max,  m <= l              (triangular)
  Zernike:            n <= n_max,  l <= l_max,  m <= l  (pencil: high radial, low angular)

Field components
----------------
B = grad(Phi), giving (derivation in CLAUDE.md):

  Bz   = (1/r_scale) * [D_extra * cos(theta) * P_l^m + (R_n^l/rho) * (l+m) * P_{l-1}^m ] * phi_factor
  Br   = (1/r_scale) / sin(theta) * [D_extra * sin^2(theta) * P_l^m
                                     + (R_n^l/rho) * (l*P_l^m - (l+m)*cos(theta)*P_{l-1}^m)] * phi_factor
  Bphi = (1/r_scale) * (R_n^l/rho) * m * P_l^m / sin(theta) * phi_factor_perp

where  D_extra = dR_n^l/drho - l*(R_n^l/rho)
               = 4*rho^(l+1) * (k + l + 3/2)/2 * P_{k-1}^{(1, l+3/2)}(2*rho^2 - 1)

Note: for n=l (k=0), D_extra=0 and all formulas reduce exactly to harmonic_basis.py.

Physical note
-------------
For k > 0, the Zernike scalar potential does not satisfy Laplace's equation, so the
resulting B field does not rigorously satisfy div(B) = 0 everywhere.  For TOSCA data
fitting (a smoothing/interpolation task) this is acceptable; for a physics-constrained
NMR calibration fit the standard harmonic basis (n=l modes only) should be used.

Reference: CLAUDE.md section "Proposed B-field Calibration Strategy", and
  Novotni & Klein, "3D Zernike Descriptors", SOLID 2003.
"""

import numpy as np
from scipy.special import lpmv, eval_jacobi


# ---------------------------------------------------------------------------
# Parameter list
# ---------------------------------------------------------------------------

def param_list(n_max, l_max=None):
    """Return list of (n, l, m, cs) tuples for all Zernike modes up to n_max.

    Parameters
    ----------
    n_max : int   maximum total degree  (controls radial resolution)
    l_max : int   maximum angular degree (controls max m; default = n_max)

    Constraints per mode:
      n >= 1              (n=0 is the gauge mode Phi=const, excluded)
      0 <= l <= min(n, l_max)
      n - l  even
      0 <= m <= l
      cs = 'c'  (cosine phi-dependence)
      cs = 's'  (sine   phi-dependence, m > 0 only)

    For n=l (k=0) these are the standard harmonic modes; for n > l they add
    extra radial depth at the same angular order.
    """
    if l_max is None:
        l_max = n_max
    params = []
    for n in range(1, n_max + 1):
        for l in range(0, min(n, l_max) + 1):
            if (n - l) % 2 != 0:
                continue
            params.append((n, l, 0, 'c'))
            for m in range(1, l + 1):
                params.append((n, l, m, 'c'))
                params.append((n, l, m, 's'))
    return params


def n_params(n_max, l_max=None):
    """Number of Zernike modes for given (n_max, l_max)."""
    return len(param_list(n_max, l_max))


# ---------------------------------------------------------------------------
# Radial functions
# ---------------------------------------------------------------------------

def _plm(l, m, cos_theta):
    """Associated Legendre P_l^m(cos_theta); zero for l < 0."""
    if l < 0:
        return np.zeros_like(cos_theta)
    return lpmv(m, l, cos_theta)


def _radial(rho, n, l):
    """Compute (R_n^l, D_extra) vectorised over rho.

    R_n^l(rho) = rho^l * P_k^{(0, l+0.5)}(2*rho^2 - 1)   [k = (n-l)//2]

    D_extra(rho) = dR_n^l/drho - l * R_n^l / rho
                 = 4*rho^(l+1) * (k+l+1.5)/2 * P_{k-1}^{(1, l+1.5)}(2*rho^2 - 1)

    D_extra is zero for k=0 (n=l), recovering the harmonic case.
    """
    k = (n - l) // 2
    x = 2.0 * rho**2 - 1.0

    Pk = eval_jacobi(k, 0.0, l + 0.5, x)

    with np.errstate(invalid='ignore'):
        rl = rho**l if l > 0 else np.ones_like(rho)
    Rn_l = rl * Pk

    if k >= 1:
        Pk1 = eval_jacobi(k - 1, 1.0, l + 1.5, x)
        D_extra = 4.0 * rho * rl * (0.5 * (k + l + 1.5)) * Pk1
    else:
        D_extra = np.zeros_like(rho)

    return Rn_l, D_extra


# ---------------------------------------------------------------------------
# Field component basis functions
# ---------------------------------------------------------------------------

def bz_basis(n, l, m, cs, r, phi, z, r_scale=1.0, z0=0.0):
    """Bz component of Zernike scalar-potential mode (n, l, m, cs).

    For n=l: identical to harmonic_basis.bz_basis(l, m, cs, ...).
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float) - z0

    R   = np.sqrt(r**2 + z**2)
    rho = R / r_scale

    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = np.where(R > 0, z / R, np.sign(z))

    Rn_l, D_extra = _radial(rho, n, l)

    # R_n^l / rho: for l=0 and k>0 this would diverge at rho=0, but the
    # coefficient (l+m) is zero for l=m=0, so the term vanishes.
    with np.errstate(invalid='ignore', divide='ignore'):
        Rn_l_over_rho = np.where(rho > 0, Rn_l / rho, 0.0)

    plm  = _plm(l,     m, cos_theta)
    plm1 = _plm(l - 1, m, cos_theta)  # zero for l=0

    phi_factor = np.cos(m * phi) if cs == 'c' else np.sin(m * phi)

    return (1.0 / r_scale) * (
        D_extra * cos_theta * plm
        + Rn_l_over_rho * (l + m) * plm1
    ) * phi_factor


def br_basis(n, l, m, cs, r, phi, z, r_scale=1.0, z0=0.0):
    """Br (cylindrical) component of Zernike scalar-potential mode (n, l, m, cs).

    For n=l: identical to harmonic_basis.br_basis(l, m, cs, ...).
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float) - z0

    R   = np.sqrt(r**2 + z**2)
    rho = R / r_scale

    with np.errstate(invalid='ignore', divide='ignore'):
        sin_theta = np.where(R > 0, r / R, 0.0)
        cos_theta = np.where(R > 0, z / R, np.sign(z))

    Rn_l, D_extra = _radial(rho, n, l)

    with np.errstate(invalid='ignore', divide='ignore'):
        Rn_l_over_rho = np.where(rho > 0, Rn_l / rho, 0.0)

    plm  = _plm(l,     m, cos_theta)
    plm1 = _plm(l - 1, m, cos_theta)

    numerator = (
        D_extra * sin_theta**2 * plm
        + Rn_l_over_rho * (l * plm - (l + m) * cos_theta * plm1)
    )

    safe = sin_theta > 1e-12
    br   = np.where(safe,
                    (1.0 / r_scale) * numerator / np.where(safe, sin_theta, 1.0),
                    0.0)

    phi_factor = np.cos(m * phi) if cs == 'c' else np.sin(m * phi)
    return br * phi_factor


def bphi_basis(n, l, m, cs, r, phi, z, r_scale=1.0, z0=0.0):
    """Bphi component of Zernike scalar-potential mode (n, l, m, cs).

    Zero for m=0.  For n=l: identical to harmonic_basis.bphi_basis(l, m, cs, ...).
    """
    if m == 0:
        return np.zeros_like(np.asarray(r, dtype=float))

    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float) - z0

    R   = np.sqrt(r**2 + z**2)
    rho = R / r_scale

    with np.errstate(invalid='ignore', divide='ignore'):
        sin_theta = np.where(R > 0, r / R, 0.0)
        cos_theta = np.where(R > 0, z / R, np.sign(z))

    Rn_l, _ = _radial(rho, n, l)

    with np.errstate(invalid='ignore', divide='ignore'):
        Rn_l_over_rho = np.where(rho > 0, Rn_l / rho, 0.0)

    plm  = _plm(l, m, cos_theta)

    safe = sin_theta > 1e-12
    base = np.where(safe,
                    (1.0 / r_scale) * Rn_l_over_rho * m * plm
                    / np.where(safe, sin_theta, 1.0),
                    0.0)

    phi_factor = -np.sin(m * phi) if cs == 'c' else np.cos(m * phi)
    return base * phi_factor


# ---------------------------------------------------------------------------
# Design matrix and field evaluation
# ---------------------------------------------------------------------------

def build_design_matrix(r, phi, z, n_max, l_max=None,
                        components=('Bz',), r_scale=None, z0=0.0):
    """Build the Zernike design matrix.

    Parameters
    ----------
    r, phi, z   : 1-D arrays  cylindrical coordinates [same units as r_scale]
    n_max       : int  maximum total degree  (radial resolution)
    l_max       : int  maximum angular degree (default = n_max)
    components  : tuple of str  ('Bz', 'Br', 'Bphi')
    r_scale     : float  normalisation radius; if None uses max(R) of input points
    z0          : float  axial origin offset

    Returns
    -------
    A       : ndarray (n_pts * n_comp, n_pars)
    params  : list of (n, l, m, cs) tuples
    r_scale : float  (same value passed back for use in eval_field)
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    if r_scale is None:
        zp = z - z0
        r_scale = float(np.max(np.sqrt(r**2 + zp**2)))
        if r_scale == 0.0:
            r_scale = 1.0

    params  = param_list(n_max, l_max)
    n_pts   = len(r)
    n_comp  = len(components)
    n_par   = len(params)

    A = np.zeros((n_pts * n_comp, n_par))

    comp_funcs = {'Bz': bz_basis, 'Br': br_basis, 'Bphi': bphi_basis}

    for ic, comp in enumerate(components):
        func   = comp_funcs[comp]
        row0   = ic * n_pts
        for ip, (nv, lv, mv, csv) in enumerate(params):
            A[row0:row0 + n_pts, ip] = func(nv, lv, mv, csv,
                                             r, phi, z,
                                             r_scale=r_scale, z0=z0)

    return A, params, r_scale


def eval_field(coeffs, params, r, phi, z,
               components=('Bz',), r_scale=1.0, z0=0.0):
    """Evaluate the Zernike fit at arbitrary (r, phi, z) points.

    Parameters
    ----------
    coeffs   : 1-D array  fitted coefficients
    params   : list of (n, l, m, cs) tuples  (from param_list or build_design_matrix)
    r, phi, z: evaluation coordinates
    components, r_scale, z0 : same values used during fitting

    Returns
    -------
    dict mapping component name -> ndarray of field values [T]
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)

    comp_funcs = {'Bz': bz_basis, 'Br': br_basis, 'Bphi': bphi_basis}
    result = {}

    for comp in components:
        func = comp_funcs[comp]
        val  = np.zeros(len(r))
        for c, (nv, lv, mv, csv) in zip(coeffs, params):
            if c != 0.0:
                val += c * func(nv, lv, mv, csv, r, phi, z,
                                r_scale=r_scale, z0=z0)
        result[comp] = val

    return result
