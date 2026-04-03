# CLAUDE.md — mfs: CMS Magnetic Field Studies Framework

## Purpose

This Python framework fits the CMS solenoid magnetic field to analytic basis functions satisfying the Laplace equation and compares fit predictions against NMR probe measurements. It is used to:

1. Fit TOSCA/OPERA grid data (field maps from Slava Klyukhin) to a smooth polynomial basis
2. Compare model predictions against NMR probe measurements at 4 fixed probe positions
3. Produce residual plots, coefficient plots, and correlation matrices
4. Provide the coefficient set for the proposed B-field calibration strategy (see main CLAUDE.md)

The `mfs` directory lives inside the CMSSW working area but is self-contained Python and does not require CMSSW to run. The primary upstream is `https://github.com/davidwalter2/mfs`.

---

## Setup

The scripts require scipy, numpy, matplotlib, and related packages. Always use the virtual environment:

```bash
cd /work/submit/david_w/ZMass/CMSSW_10_6_26/src/mfs
source .venv/bin/activate
# now: python, pip, etc. all use the venv
```

To recreate the venv from scratch:
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib mplhep pandas wums h5py hdf5plugin lz4
```

---

## Scripts

| Script | Purpose |
|---|---|
| `fit_field.py` | Main fitting script — loads a field grid, builds design matrix, solves least squares, saves `.npz` |
| `harmonic_basis.py` | Spherical harmonic scalar potential basis: `bz_basis`, `br_basis`, `bphi_basis`, `build_design_matrix`, `eval_field` |
| `cylindrical_basis.py` | Fourier-Bessel cylindrical basis: I_m(k r) sin/cos(k z) (poor convergence for solenoid — see below) |
| `cylindrical_bessel_basis.py` | J-Bessel + cosh/sinh basis: J_m(α r/R) cosh/sinh(α z/R) — physically motivated cylindrical basis |
| `zernike_basis.py` | Zernike-based radial × Legendre-z basis (experimental) |
| `plot_residuals.py` | Reads `.npz` fit output, generates residual maps, radial profiles, RMS tables |
| `plot_coeffs.py` | Plots fitted coefficients as 2D scatter (m vs l, or m vs n for cylindrical bases) |
| `plot_corr.py` | Plots parameter correlation matrix (Gram matrix) for any fit |
| `nmr_fit.py` | Compares fit predictions against NMR probe measurements |
| `plot_NMR_data.py` | Plots NMR time series from `data/nmr_probes/NMR_tabulated.csv` |
| `analyze_bfield.py` | Additional field analysis utilities |
| `compare_z0.py` | Compares fits with different axial origin offsets z0 |
| `inspect_radial.py` | Inspects radial profiles from a fit |
| `z0_scan.py` | Scans z0 offset values to find the magnetic centre |

---

## Grid Format

Field grids are produced by `MagneticField/Engine/test/dumpField_cfg.py` (CMSSW, requires el7 container). The output text format is:

```
# r[cm]  phi[rad]  z[cm]  x[cm]  y[cm]  Bx[T]  By[T]  Bz[T]
```

Key grid files used (stored in `MagneticField/Engine/test/` in the CMSSW tree):

| File | Grid | Points | Purpose |
|---|---|---|---|
| `field_tosca170812_sphere320.txt` | 65×36×66, r∈[0,320], z∈[±325] cm | 154,446 | **Primary fit grid** (sphere320, current best) |
| `field_tosca170812_extended.txt` | 60×36×61, r∈[0,295], z∈[±300] cm | 130,680 | Extended, near coil boundary |
| `field_tosca170812_full.txt` | 30×36×61, r∈[0,280], z∈[±300] cm | 65,880 | Full bore (older) |
| `field_tosca170812.txt` | tracker grid, 24×12×57 | ~16k | Tracker-only, low-order fits |
| `field_polyfit3d.txt` | same tracker grid, PolyFit3D model | ~16k | Compare against CMSSW parametrization |
| `field_polyfit3d_full.txt` | full domain, PolyFit3D model | ~65k | PolyFit3D over full tracker region |

The CMSSW command to generate `field_tosca170812_sphere320.txt` (inside the el7 container):
```bash
APPTAINER_BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run" \
  cmssw-el7 --command-to-run bash -c "
    cd /work/submit/david_w/ZMass/CMSSW_10_6_26/src &&
    source /cvmfs/cms.cern.ch/cmsset_default.sh &&
    eval \$(scramv1 runtime -sh) &&
    cd MagneticField/Engine/test &&
    cmsRun dumpField_cfg.py model=tosca170812 grid=sphere320
  "
```

---

## Fit Domain

### Spherical domain (sphere320) — default for spherical harmonic basis

Three combined cuts:
- **r < 290 cm**: coil conductor starts here; ∇²Φ ≠ 0 inside coil. Using r<308 degrades fit ~25%.
- **|z| < 316 cm**: solenoid volume V_1008 half-length; outside this TOSCA interpolation is unreliable.
- **R_sph < 320 cm**: spherical cut enclosing the full tracker volume (tracker envelope: r<123.3, |z|<293.5, corner R=318.3 cm).

Grid: `field_tosca170812_sphere320.txt` — 154,446 total points; **113,619 points** after cuts.

r_scale = 319.9 cm (maximum R in fitting region, used for normalization).

### Cylindrical domain — natural for cylindrical (J-Bessel) bases

- **r < 290 cm** + **|z| < 300 cm** — no sphere radius cut needed.
- Use `field_tosca170812_extended.txt` or `field_polyfit3d_full.txt` as input.

### Tracker Boundaries (from CMSSW `isDefined()`)

| Region | r limit | |z| limit | Notes |
|---|---|---|---|
| **OAE** (standard reco) | < 115 cm | < 280 cm | Default field parametrization |
| **PolyFit3D** (CVH refit) | < 190 cm | < 350 cm | Plus corner cut: |z|+2.5r < 670 cm |

---

## Basis Types and Convergence

### 1. Spherical Harmonic (Maroussov) — `--basis harmonic` (default)

```
Phi = sum_{l,m}  R^l P_l^m(cos theta) [A_lm cos(m phi) + B_lm sin(m phi)]
```

B = ∇Φ, modes are polynomials in (r, z) — the natural Laplace solution for a solenoid. Field variation is primarily in polar angle θ (captured by Legendre P_l^m), matching the solenoid's angular structure.

**Parameter count:** (l_max+1)² − 1 (l=0 excluded as gauge mode)

| l_max | Params | Bz RMS (tracker) | Cond. # |
|---|---|---|---|
| 6 | 48 | ~0.28 mT | <10 |
| 8 | 80 | ~0.16 mT | <10 |
| 12 | 168 | ~0.12 mT | <10 |
| 18 | 360 | 0.095 mT | 7.5 |

Full rank, excellent conditioning. **Recommended basis.**

### 2. Fourier-Bessel Cylindrical — `--basis cylindrical`

```
Phi = I_m(k_n r) sin/cos(k_n z)  with  k_n = n*pi/L
```

Uses oscillatory sin/cos in z — wrong for a monotonic solenoid field. Converges ~200× slower than spherical harmonics. n=10, m=1 (61 params): Bz RMS ~22.8 mT on PolyFit3D data (vs 0.28 mT for spherical harmonics at same parameter count).

**Not recommended** for practical fitting. Only useful as a mathematical cross-check.

### 3. J-Bessel + cosh/sinh — `--basis bessel`

```
Phi = J_m(alpha_{m,n} r / R) * cosh or sinh(alpha_{m,n} z / R) * cos/sin(m phi)
```

where `alpha_{m,n}` = n-th positive zero of J_m (Dirichlet BC: Phi=0 at r=R). Uses hyperbolic (monotonic) z-functions instead of oscillatory sin/cos — physically motivated since the solenoid Bz is monotonic. **10× better** than Fourier-Bessel at same parameter count.

Mode structure (same as Fourier-Bessel):
- n=0, m=0: uniform Bz (constant background)
- `cz='sym'` → cosh(k z), `cz='anti'` → sinh(k z)
- Total modes: `1 + 2*n_max*(1 + 2*m_max)`

**Key limitation:** cosh(k z_max) ~ exp(k z_max) grows exponentially for high n → condition number grows rapidly with n_max. Column normalization handles absolute scale but relative conditioning remains poor.

| Config | Params | Bz RMS (PolyFit3D) | Cond. # |
|---|---|---|---|
| n=10, m=1, R=290 | 61 | 1.34 mT | 731 |
| n=10, m=1, R=320 | 61 | 0.75 mT | 2084 |
| n=15, m=1, R=320 | 91 | 0.49 mT | 85k |

**vs spherical harmonics at same parameter count:** l_max=5 (48 params) → Bz 0.28 mT, cond ~7. The J-Bessel basis needs ~91 params to approach this accuracy at much higher condition number.

**Radial scale R:** R=290 cm (= r_max of fit domain) is physically motivated — J_m(k R)=0 means the Bessel zero falls exactly at the domain boundary. R=320 gives slightly better RMS but worse conditioning.

**Fit domain:** Use pure cylindrical (r<290, |z|<300), no sphere radius cut needed.

### 4. Zernike — `--basis zernike`

Zernike radial polynomials × Legendre in z. Experimental. Results not conclusive yet; stored in `data/fitresults/` with `zernike_` prefix.

### Basis Comparison (PolyFit3D data, r<290, |z|<300, excl vol-boundaries, m=1)

| Basis | Params | Bz RMS | Br RMS | Cond |
|---|---|---|---|---|
| Fourier-Bessel (I_m+sin/cos), n=10 | 61 | 12.4 mT | 10.0 mT | 7 |
| J-Bessel+cosh/sinh, n=10, R=290 | 61 | 1.34 mT | 1.30 mT | 731 |
| J-Bessel+cosh/sinh, n=10, R=320 | 61 | 0.75 mT | 0.76 mT | 2084 |
| J-Bessel+cosh/sinh, n=15, R=320 | 91 | 0.38 mT | 0.49 mT | 85k |
| **Spherical harmonic l=18 lphi=5** | **48** | **0.06 mT** | **0.12 mT** | — |

The Fourier-Bessel and J-Bessel bases capture radial variation via Bessel zeros, but the solenoid's interesting structure is in polar angle θ (Legendre polynomials), not radius r. This fundamental mismatch limits cylindrical bases regardless of parameter count.

---

## Spherical Harmonic Truncation Schemes

### Standard triangular truncation

```bash
python fit_field.py ... --lmax 18
# all (l,m) with l<=18
```

### Asymmetric phi-truncation (`--lmax-phi`)

Reduces the maximum m for all l to save parameters on phi-harmonics:

```bash
python fit_field.py ... --lmax 18 --lmax-phi 5
# uses m<=5 for all l, but full l up to 18
```

### Diamond truncation (`--nmax-sum`)

Restricts l + m ≤ n_max_sum:

```bash
python fit_field.py ... --lmax 12 --nmax-sum 12
# excludes high-l, high-m corners
```

### Custom per-l truncation (`--m-max-per-l`)

Specify max m for each l individually. Format: `"l:m,l2:m2,lrange-lend:m"`:

```bash
python fit_field.py ... --lmax 18 --m-max-per-l "1:1,2:2,3:3,4:4,5:5,6:1,7-18:0"
```

### Best 50-parameter prescription

**`lphi5-base + l=6,m=1`**: full triangle to l=5 (48 params) plus l=6,m=1 (2 params) = 50 params total.

CLI: `--lmax 18 --m-max-per-l "1:1,2:2,3:3,4:4,5:5,6:1,7-18:0"`

Results (PolyFit3D, sphere320 domain, excl vol-boundaries):
- **Bz RMS: 0.055 mT** (tracker region r<115, |z|<280)
- **Br RMS: 0.095 mT**
- Outperforms full l_max=5 (48 params) meaningfully; adding l=6,m=3+ gives diminishing returns

For TOSCA data (sphere320): `data/fitresults/tosca170812_sphere320_coeffs_lmax18_custom50.npz`
For PolyFit3D: `data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.npz`

---

## fit_field.py CLI Reference

```bash
python fit_field.py -i <grid.txt> [options]
```

### Basis selection

| Option | Values | Description |
|---|---|---|
| `--basis` | `harmonic` (default), `cylindrical`, `bessel`, `zernike` | Which basis to use |

### Harmonic basis options

| Option | Default | Description |
|---|---|---|
| `--lmax` | 18 | Maximum l (spherical degree) |
| `--lmax-phi` | None | Maximum m across all l (asymmetric phi-truncation) |
| `--nmax-sum` | None | Diamond truncation: keep l+m ≤ nmax-sum |
| `--m-max-per-l` | None | Custom per-l m_max, e.g. `"1:1,2:2,3:3,4:4,5:5,6:1,7-18:0"` |
| `--rscale` | auto | Normalisation radius r_scale [cm] |
| `--z0` | 0 | Axial origin offset [cm] |

### Cylindrical/Bessel basis options

| Option | Default | Description |
|---|---|---|
| `--nmax` | 10 | Maximum n (wavenumber index) |
| `--mmax` | 1 | Maximum m (phi harmonic) |
| `--L` | auto | Half-period L for Fourier-Bessel [cm] (= z_max) |
| `--rscale` | auto | Radial scale R for J-Bessel [cm] (= r_max; set explicitly to 290 or 320) |

### Domain cuts

| Option | Default | Description |
|---|---|---|
| `--rmax` | 290 | Cylindrical r cut [cm] |
| `--zmax` | 316 | \|z\| cut [cm] |
| `--rmax-sphere` | None | Spherical R cut [cm] (use 320 for sphere320 domain) |
| `--rmin-sphere` | None | Inner sphere exclusion [cm] |
| `--exclude-vol-boundaries` | off | Exclude ±12 cm around CMSSW volume z-boundaries |

### Regularization

| Option | Default | Description |
|---|---|---|
| `--tikhonov` | 0 | Tikhonov λ (0 = no regularization) |
| `--tikhonov-power` | 2 | Spectral power s; penalty scales as [l(l+1)]^s |

### Output

| Option | Default | Description |
|---|---|---|
| `--components` | `Bz Br` | Which field components to fit (space-separated; add `Bphi` for 3-component) |
| `--output` | auto-stem | Output `.npz` path (auto prefix: `data/fitresults/`) |

### Common invocations

```bash
# Best harmonic fit (sphere320 domain)
python fit_field.py -i ../MagneticField/Engine/test/field_tosca170812_sphere320.txt \
  --lmax 18 --rmax 290 --zmax 316 --rmax-sphere 320 \
  --components Bz Br Bphi --exclude-vol-boundaries

# Best 50-parameter (custom) fit
python fit_field.py -i ../MagneticField/Engine/test/field_tosca170812_sphere320.txt \
  --lmax 18 --rmax 290 --zmax 316 --rmax-sphere 320 \
  --components Bz Br Bphi --exclude-vol-boundaries \
  --m-max-per-l "1:1,2:2,3:3,4:4,5:5,6:1,7-18:0"

# J-Bessel fit (cylindrical domain, R=290)
python fit_field.py -i ../MagneticField/Engine/test/field_polyfit3d_full.txt \
  --basis bessel --nmax 10 --mmax 1 --rscale 290 \
  --rmax 290 --zmax 300 --components Bz Br --exclude-vol-boundaries
```

---

## Key Fit Results

### Spherical harmonic basis (TOSCA, sphere320, lmax=18, no regularization)

| Region | Bz RMS | Br RMS | Bphi RMS |
|---|---|---|---|
| Global (113,619 pts) | **0.95 mT** | 0.99 mT | 0.013 mT |
| Tracker (r<115, |z|<280) | **0.095 mT** | ~0.1 mT | — |
| Near boundary (R>270 cm) | **~1.82 mT** | ~1.82 mT | — |

Condition number: 7.5. No regularization needed.

### Effect of spectral Tikhonov regularization

Penalty for mode (l,m) scales as [l(l+1)]^s. Best: λ=1e-3, s=3 → tracker RMS 0.067 mT (small improvement). Boundary residuals unchanged at ~1.82 mT regardless of λ — confirmed irreducible TOSCA artifacts (volume interpolation kinks, not polynomial overfitting).

### Effect of volume boundary exclusion (`--exclude-vol-boundaries`)

Excludes ±12 cm around z = ±126.8, ±142.3, ±181.3 cm (CMSSW primitive volume z-boundaries).

| Fit | Tracker Bz RMS | Boundary Bz RMS |
|---|---|---|
| baseline (lmax=18, no reg) | 0.092 mT | 1.82 mT |
| + exclude-vol-boundaries | **0.073 mT** | ~1.85 mT |

### CMSSW Primitive Volume Boundaries (z-kinks)

The CMSSW field model uses 11,136 primitive volumes with piecewise bilinear interpolation. Gradient discontinuities (kinks) at z-boundaries are visible in residual maps:

| z boundary (cm) | Amplitude (Bz) | Physical origin |
|---|---|---|
| ±126.8 cm | ~0.07–0.10 mT | Coil module boundary |
| ±142.3 cm | ~0.07–0.10 mT | Thermal shield/vacuum vessel |
| ±181.3 cm | ~0.07–0.10 mT | Cryogenic chimney (v1103_071212) |

### Effect of Gauss-Legendre weights (`--gl-weights`)

**Does not help.** Tracker RMS worsens 0.096→0.145 mT; boundary RMS worsens 1.82→1.99 mT. The oscillations are Runge phenomenon at the l_max sphere boundary — intrinsic to basis truncation, not a sampling artifact.

---

## Visualization Scripts

### Residual plots

```bash
python plot_residuals.py \
  --fit data/fitresults/tosca170812_sphere320_coeffs_lmax18_all3.npz \
  --grid ../MagneticField/Engine/test/field_tosca170812_sphere320.txt \
  --out residuals_lmax18.pdf
```

Grid cuts are read automatically from the `.npz`. Draws OAE and PolyFit3D tracker boundary rectangles, per-shell radial RMS profiles, and (r,z)/(phi,z) residual maps.

Supports `basis_type` values: `harmonic`, `cylindrical`, `bessel`, `zernike`.

### Coefficient plots

```bash
python plot_coeffs.py \
  --fit data/fitresults/tosca170812_sphere320_coeffs_lmax18_all3.npz \
  --out coeffs.pdf
```

- Harmonic: 2D scatter, x = m (phi order), y = l (degree), marker size ∝ |coefficient|
- Cylindrical/Bessel: x = n (positive = z-symmetric cosh/sin, negative = z-antisymmetric sinh/cos), y = m

X-axis range auto-scales to the actual m range in the fit (important for custom truncations that don't use all m values up to l_max).

### Correlation matrix

```bash
python plot_corr.py \
  --fit data/fitresults/tosca170812_sphere320_coeffs_lmax18_all3.npz \
  --grid ../MagneticField/Engine/test/field_tosca170812_sphere320.txt \
  --out corr.pdf
```

Computes the Gram matrix G = A^T A of the design matrix; normalizes to correlation. Block boundaries drawn at each new l (harmonic) or n (cylindrical/bessel) group.

---

## Output Files: `data/fitresults/`

All `.npz` fit output files are stored in `data/fitresults/`. The `data/nmr_probes/` subdirectory holds NMR time-series data.

### NPZ File Format

All fits share common keys:

| Key | Description |
|---|---|
| `basis_type` | `'harmonic'`, `'cylindrical'`, `'bessel'`, or `'zernike'` |
| `coeffs` | Fitted coefficients, shape `(n_params,)` |
| `components` | Which field components were fitted (`['Bz', 'Br']` etc.) |
| `col_norms` | Column norms used for internal scaling |
| `tikhonov` | Regularization strength λ (0 = none) |
| `rms_Bz/Br/Bphi` | RMS residuals per component [T] |
| `condition_number` | Condition number of the scaled design matrix |
| `rmax_cm` | Cylindrical r cut [cm] |
| `zmax_cm` | \|z\| cut [cm] |
| `rmax_sphere_cm` | Spherical R cut [cm] (NaN if not applied) |
| `rmin_sphere_cm` | Inner sphere exclusion [cm] (NaN if not applied) |
| `exclude_vol_boundaries` | Bool — whether vol-boundary exclusion was applied |

**Harmonic-specific keys:**

| Key | Description |
|---|---|
| `params` | Parameter labels as strings `"l{l}_m{m}_{cs}"` |
| `l_max` | Maximum harmonic degree l |
| `l_max_phi` | Max m across all l (NaN if triangular) |
| `n_max_sum` | Diamond truncation threshold (NaN if unused) |
| `m_max_per_l_str` | Custom per-l truncation string (NaN if unused) |
| `r_scale` | Normalisation radius [cm] |
| `z0` | Axial origin offset [cm] |

**Cylindrical/Bessel-specific keys:**

| Key | Description |
|---|---|
| `params` | Mode labels as strings `"n{n}_m{m}_{cphi}_{cz}"` |
| `n_max` | Maximum n (wavenumber index) |
| `m_max` | Maximum m (phi harmonic) |
| `L` | Half-period [cm] (Fourier-Bessel only) |
| `R` | Radial scale [cm] (J-Bessel only) |

### Naming Conventions

Harmonic: `{grid_stem}_coeffs_lmax{N}[_{tags}].npz`

| Tag | Meaning |
|---|---|
| `_all3` | Fit to Bz + Br + Bphi simultaneously |
| `_noreg` | No Tikhonov regularization (default when not specified) |
| `_s{k}` | Spectral Tikhonov with power s=k (and λ=1e-3) |
| `_excl` | CMSSW volume z-boundary exclusion applied |
| `_custom` or `_custom50` | Custom per-l truncation (50-parameter prescription) |
| `_lphi{m}` | Asymmetric phi-truncation with l_max_phi=m |
| `_diamond{n}` | Diamond truncation with nmax_sum=n |
| `_rmax{R}` | Non-default r cut |
| `_z0_{d}mm` | Non-zero axial origin offset |
| `_gl` | Gauss-Legendre quadrature weights (tested, worsens results — avoid) |

Cylindrical: `{grid_stem}_cyl_n{N}m{M}.npz`
Bessel: `{grid_stem}_coeffs_bessel_nmax{N}_mmax{M}.npz`
Zernike: `{grid_stem}_zernike_nmax{N}_lmax{L}[_excl][_rmin1].npz`

**Current bests:**
- TOSCA, sphere320, lmax=18: `tosca170812_sphere320_coeffs_lmax18_all3.npz` (0.095 mT tracker)
- TOSCA, sphere320, 50 params: `tosca170812_sphere320_coeffs_lmax18_custom50.npz`
- PolyFit3D, 50 params: `polyfit3d_full_coeffs_lmax18_custom50.npz` (0.055 mT tracker)
- J-Bessel, n=10, R=290: `field_polyfit3d_full_coeffs_bessel_nmax10_mmax1.npz` (1.34 mT — much worse)

---

## NMR Probe Data

### Probe Positions (Run 2, CMS global coordinates)

| Probe | X (cm) | Y (cm) | Z (cm) | r (cm) | phi (rad) | B_meas 2017 (T) |
|---|---|---|---|---|---|---|
| A | −206.345 | −205.870 | −0.6 | 291.5 | −2.357 | 3.92165 |
| E | −206.345 | +205.870 | +0.6 | 291.5 | +2.357 | 3.92118 |
| C | +64.25 | +10.517 | −283.5 | 65.1 | +0.162 | 3.65630 |
| D | +64.25 | +10.517 | +283.1 | 65.1 | +0.162 | 3.65990 |

Probes A/E: at the solenoid inner wall (r~291 cm), z≈0 (midplane), ~114° separation in phi — constrain phi-asymmetry at the coil boundary. Probes C/D: inside the **tracker volume** at opposite z-endcaps, same (r, phi).

NMR precision: (5.2 ± 1.3) × 10⁻⁵ T.

### TOSCA Residuals vs 2017 NMR Measurements

| Probe | TOSCA − NMR (mT) | Notes |
|---|---|---|
| A | +2.91 | midplane, large r |
| E | +2.52 | midplane, large r |
| C | +1.59 | endcap, small r |
| D | +1.91 | endcap, small r |

Residuals are not uniform: A ≠ E (phi-asymmetry ~0.2 mT) and D > C (z-asymmetry ~0.3 mT). Dominant correction is l=1 m=0 (overall scale ~+2 mT), with l=2 and low-order phi-asymmetric modes needed for residual structure.

### Position Uncertainty (important systematic for C/D)

The tracker is shifted **2 mm toward −Z** in CMS global coordinates. With ∂Bz/∂z ~ 0.2–0.3 T/m at the endcap, a 2 mm Z uncertainty → **0.4–0.6 mT field uncertainty** — an order of magnitude larger than NMR precision. Probe position must be floated as a free parameter in any NMR-constrained fit.

### Data Files

```
data/nmr_probes/NMR_tabulated.csv               # time series, all 4 probes, 2006-2025, at 18164 A
data/nmr_probes/field_results_170812_run2.txt    # TOSCA predictions at probe locations
data/nmr_probes/field_results_polyfit2d.txt      # Maroussov 2D parametrization predictions
data/nmr_probes/field_results_polyfit3d.txt      # Maroussov 3D parametrization predictions
```

Note: C/D NMR readings are only available for 2017 within Run 2. A/E data covers a longer period.

---

## Next Step: NMR Constraints in the Fit

The ~1.82 mT boundary residuals occur near the coil (r~290 cm), exactly where NMR probes A/E are located. NMR measurements with 0.05 mT precision can anchor the fit solution in this region.

**Why non-NMR approaches don't solve this:**
- Tikhonov regularization (all variants): boundary residuals unchanged at ~1.82 mT
- Volume boundary exclusion: improves clean-tracker RMS but boundary residuals remain
- GL-weighted sampling: definitively tested and fails — worsens tracker RMS
- The oscillations are Runge phenomenon at the l_max sphere boundary, intrinsic to basis truncation

**NMR constraint implementation plan** (not yet done):
1. Load measured field values from `NMR_tabulated.csv` (2017 run, 18164 A)
2. Evaluate TOSCA prediction at each probe (from `field_results_170812_run2.txt`) — difference is the NMR residual
3. Append 4 NMR constraint rows to the design matrix, weighted by σ_TOSCA/σ_NMR ~ 1.82 mT / 0.05 mT ~ 36
4. Re-solve the least-squares problem — anchors near-boundary modes to NMR data
5. Float probe C/D positions (Z offset ±few mm) as nuisance parameters

**Potential improvements for J-Bessel basis** (if revisiting):
- Neumann BC: zeros of J_m' instead of J_m (avoids forcing field to vanish at r=R) — `scipy.special.jnp_zeros(m, n)`
- Mixed polynomial + Bessel: low-order polynomial modes as backbone, Bessel for higher-frequency content
- Per-m different n_max: m=0 (axisymmetric) may need more Bessel zeros than high-m modes
- Tikhonov regularization at n=15+: condition ~85k is still usable with regularization

---

## TOSCA Model Context

The TOSCA (v170812) model covers the full CMS detector (18m × 120m domain, 8.7M mesh nodes). Post-processed into 11,136 primitive volumes for fast lookup + bilinear interpolation in CMSSW. Accuracy: **~0.1%** inside the tracker, ~3% in yoke steel.

The **missing turn** in coil module CB-2 (z = −3.77 m) shifts the axial field ~15–20 mm toward +Z. TOSCA models this: OPERA predicts D > C by 3.37 mT vs. measured 3.6 mT — good agreement. The dominant NMR residual (~1.7–2.7 mT) is the overall scale, not the missing turn.

Key references:
- Amapane, Klyukhin. "Development of the CMS Magnetic Field Map." Symmetry 2023, 15, 1030. arXiv:2401.01913
- Klyukhin. "Design and Description of the CMS Magnetic System Model." Symmetry 2021, 13, 1052
- Maroussov PhD thesis, 2008 (local: `/work/submit/david_w/Documents/theses/maroussov.pdf`)
- Field-0045 (missing turn note, local: `/work/submit/david_w/Documents/notes/TeX-Field-0045.pdf`)
