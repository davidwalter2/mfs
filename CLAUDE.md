# CLAUDE.md — mfs: CMS Magnetic Field Studies Framework

## Purpose

This Python framework fits the CMS solenoid magnetic field to the **Maroussov spherical harmonic scalar potential** basis and compares the fit against NMR probe measurements. It is used to:

1. Fit TOSCA/OPERA grid data (field maps from Slava Klyukhin) to a smooth polynomial basis
2. Compare model predictions against NMR probe measurements at 4 fixed probe positions
3. Produce residual plots and evaluate fit quality in the tracker volume
4. Provide the coefficient set for the proposed B-field calibration strategy (see below)

The `mfs` directory lives inside the CMSSW working area but is self-contained Python and does not require CMSSW to run. The main CMSSW repository (upstream of this tree) is `https://github.com/davidwalter2/mfs`.

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
| `fit_field.py` | Main fitting script — loads a field grid, builds the design matrix, solves least squares, saves `.npz` |
| `harmonic_basis.py` | Spherical harmonic basis functions: `bz_basis`, `br_basis`, `bphi_basis`, `build_design_matrix`, `eval_field` |
| `cylindrical_basis.py` | Fourier-Bessel cylindrical basis (alternative, worse convergence for solenoid fields) |
| `plot_residuals.py` | Reads `.npz` fit output, generates residual maps, radial profiles, RMS tables |
| `nmr_fit.py` | Compares fit predictions against NMR probe measurements |
| `plot_NMR_data.py` | Plots the NMR time series from `data/nmr_probes/NMR_tabulated.csv` |
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
| `field_tosca170812_full.txt` | 30×36×61, r∈[0,280], z∈[±300] cm | 65,880 | Full bore (older) |
| `field_tosca170812_extended.txt` | 60×36×61, r∈[0,295], z∈[±300] cm | 130,680 | Extended, near coil boundary |
| `field_tosca170812.txt` | tracker grid, 24×12×57 | ~16k | Tracker-only, low-order fits |
| `field_polyfit3d.txt` | same tracker grid, PolyFit3D model | ~16k | Compare against CMSSW parametrization |

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

## Physics Background: Maroussov Scalar Potential Basis

### Scalar Potential (Laplace Equation)

Inside the solenoid bore (no currents for r < 290 cm), the magnetic field satisfies ∇×B = 0 and ∇·B = 0, so B = ∇Φ where ∇²Φ = 0 (Laplace equation). The solution in spherical coordinates is:

```
Phi(r, phi, z) = sum_{l=1}^{l_max} sum_{m=0}^{l}
                   [ A_{l,m} cos(m*phi) + B_{l,m} sin(m*phi) ] * R^l * P_l^m(cos(theta))
```

where R = sqrt(r² + z²), cos(θ) = z/R, and P_l^m are the associated Legendre polynomials (scipy convention, no Condon-Shortley phase). **l=0 is excluded** because Φ = constant gives zero field (gauge mode; makes design matrix rank-deficient without regularization).

### Field Components

```
Bz   = (l+m) (R/r_scale)^{l-1} / r_scale * P_{l-1}^m(cos(theta)) * phi_factor
Br   = (R/r_scale)^{l-1} / r_scale * [l P_l^m - cos(theta)*(l+m)*P_{l-1}^m] / sin(theta) * phi_factor
Bphi = (m/r) (R/r_scale)^l * P_l^m(cos(theta)) * phi_factor_perp
```

These are implemented in `harmonic_basis.py` with careful handling of singularities at r=0 and on the z-axis.

### Relation to Cylindrical Polynomials

The spherical harmonics R^l P_l^m(cosθ) are polynomials in (r, z) when expressed in cylindrical coordinates — **not** functions involving Bessel functions or transcendentals. This makes them the natural basis for a smooth solenoid field. The Fourier-Bessel cylindrical basis (in `cylindrical_basis.py`) converges much more slowly for the CMS solenoid and is not recommended.

### Parameter Count

- Total parameters for l_max=N: **(N+1)² - 1** (l=0 excluded)
  - l_max=6: 48 params
  - l_max=8: 80 params
  - l_max=12: 168 params
  - l_max=18: 360 params

### Z-Parity Selection Rules

For a z-symmetric solenoid (as a good approximation), Bz is even in z and Br is odd in z. Only **odd-l modes** contribute to even-Bz (the dominant component). Even-l modes contribute near-zero for the CMS solenoid and add noise rather than signal.

### Reference

Maroussov, V. "Fit to an Analytic Form of the Measured Central CMS Magnetic Field." PhD thesis, Purdue University, 2008. Local copy: `/work/submit/david_w/Documents/theses/maroussov.pdf`. The `BFit3D` C++ implementation in `MagneticField/ParametrizedEngine/` is the direct CMSSW integration of this thesis work.

---

## Fit Domain: sphere320 (Current Best)

The primary fitting domain is defined by three combined cuts applied at fit time:

- **r < 290 cm**: hard physical boundary — coil conductor begins here; ∇²Φ ≠ 0 inside the coil. Using r < 308 cm degrades fit by ~25%.
- **|z| < 316 cm**: V_1008 (solenoid volume) half-length; outside this TOSCA interpolation is unreliable.
- **R_sph < 320 cm**: spherical cut that encompasses the full tracker volume (tracker envelope: r < 123.3 cm, |z| < 293.5 cm, corner R_sph = 318.3 cm < 320 cm).

Grid: `field_tosca170812_sphere320.txt` — 154,446 total points; **113,619 points** after cuts.

r_scale = 319.9 cm (maximum R in the fitting region, used for numerical normalization).

These cuts are stored in the output `.npz` files and read back automatically by `plot_residuals.py`.

### Tracker Boundaries (from CMSSW `isDefined()`)

| Region | r limit | |z| limit | Notes |
|---|---|---|---|
| **OAE** (standard reco) | < 115 cm | < 280 cm | Default field parametrization |
| **PolyFit3D** (CVH refit) | < 190 cm | < 350 cm | Plus corner cut: |z|+2.5r < 670 cm |

---

## Key Fit Results (lmax=18, sphere320, no regularization)

| Region | Bz RMS | Br RMS | Bphi RMS |
|---|---|---|---|
| Global (113,619 pts) | **0.95 mT** | 0.99 mT | 0.013 mT |
| Tracker (r<115, |z|<280) | **0.095 mT** | ~0.1 mT | — |
| Near boundary (R>270 cm) | **~1.82 mT** | ~1.82 mT | — |

- Full rank: 360/360. Condition number: 7.5. No regularization needed for a stable fit.
- Tracker accuracy is 10× better than global average.
- Boundary residuals (~1.82 mT) are **irreducible TOSCA artifacts** — they do not decrease with stronger regularization. They reflect the volume-decomposed interpolation structure of the TOSCA model at primitive volume boundaries, not harmonic overfitting.

### Effect of Spectral Tikhonov Regularization

Penalty for mode (l,m) scales as [l(l+1)]^s. Best result: λ=1e-3, s=3 → tracker RMS 0.067 mT (small improvement). Boundary residuals unchanged at ~1.82 mT regardless of λ or s — confirmed irreducible.

```bash
python fit_field.py -i ../MagneticField/Engine/test/field_tosca170812_sphere320.txt \
  --lmax 18 --rmax 290 --zmax 316 --rmax-sphere 320 \
  --components Bz Br Bphi \
  --tikhonov 1e-3 --tikhonov-power 3 \
  --output tosca170812_sphere320_coeffs_lmax18_s3
```

### Residual Plots

```bash
python plot_residuals.py \
  --fit tosca170812_sphere320_coeffs_lmax18_noreg.npz \
  --grid ../MagneticField/Engine/test/field_tosca170812_sphere320.txt \
  --rmax-sphere 320 \
  --out residuals_lmax18_noreg.pdf
```

The plot automatically applies the same r/z/sphere cuts stored in the npz, draws OAE and PolyFit3D tracker boundary rectangles, and shows per-shell radial RMS profiles.

---

## NMR Probe Data

### Probe Positions (Run 2, CMS global coordinates)

| Probe | X (cm) | Y (cm) | Z (cm) | r (cm) | phi (rad) | B_meas 2017 (T) |
|---|---|---|---|---|---|---|
| A | −206.345 | −205.870 | −0.6 | 291.5 | −2.357 | 3.92165 |
| E | −206.345 | +205.870 | +0.6 | 291.5 | +2.357 | 3.92118 |
| C | +64.25 | +10.517 | −283.5 | 65.1 | +0.162 | 3.65630 |
| D | +64.25 | +10.517 | +283.1 | 65.1 | +0.162 | 3.65990 |

Probes A/E: at the solenoid inner wall (r~291 cm), z≈0 (midplane), separated by ~114° in phi — constrain phi-asymmetry at the coil boundary. Probes C/D: inside the **tracker volume** at opposite z-endcaps, same (r, phi).

NMR precision: (5.2 ± 1.3) × 10⁻⁵ T.

### TOSCA Residuals vs 2017 NMR Measurements (OPERA predictions)

| Probe | TOSCA − NMR (mT) | Notes |
|---|---|---|
| A | +2.91 | midplane, large r |
| E | +2.52 | midplane, large r |
| C | +1.59 | endcap, small r |
| D | +1.91 | endcap, small r |

Residuals are not uniform: A ≠ E (phi-asymmetry ~0.2 mT) and D > C (z-asymmetry ~0.3 mT). A simple scale offset cannot explain all four — motivates the spatial parameterization. Dominant correction is l=1 m=0 (overall scale ~+2 mT), with l=2 and low-order phi-asymmetric modes needed for residual structure.

### Position Uncertainty (important systematic for C/D)

The tracker is shifted **2 mm toward −Z** in CMS global coordinates. With ∂Bz/∂z ~ 0.2–0.3 T/m at the endcap, a 2 mm Z uncertainty → **0.4–0.6 mT field uncertainty** — an order of magnitude larger than NMR precision. Probe position must be floated as a free parameter in any NMR-constrained fit.

### Data Files

```
data/nmr_probes/NMR_tabulated.csv          # time series, all 4 probes, 2006-2025, at 18164 A
data/nmr_probes/field_results_170812_run2.txt   # TOSCA predictions at probe locations
data/nmr_probes/field_results_polyfit2d.txt     # Maroussov 2D parametrization predictions
data/nmr_probes/field_results_polyfit3d.txt     # Maroussov 3D parametrization predictions
```

Note: C/D NMR readings are only available for 2017 within Run 2. A/E data covers a longer period.

---

## Planned Next Step: NMR Constraints in the Fit

The ~1.82 mT boundary residuals occur near the coil (r~290 cm), exactly where the NMR probes A/E are located. NMR measurements with 0.05 mT precision can anchor the fit solution in this region.

**Implementation plan** (not yet done):
1. Load measured field values from `NMR_tabulated.csv` (2017 run, 18164 A)
2. Evaluate TOSCA prediction at each probe (from `field_results_170812_run2.txt`) — difference is the NMR residual
3. Append 4 NMR constraint rows to the design matrix, weighted by σ_TOSCA/σ_NMR ~ 1.82 mT / 0.05 mT ~ 36
4. Re-solve the least-squares problem — the fit will anchor near-boundary modes to the NMR data
5. Float probe C/D positions (Z offset ±few mm) as nuisance parameters since ∂Bz/∂z ~ 0.3 T/m makes them uncertain to ~0.5 mT

For the **residual fit** strategy (fitting δc_{l,m} corrections to the TOSCA baseline rather than the absolute field), l_max~4–6 is likely sufficient: 25–49 parameters for small corrections.

---

## TOSCA Model Context

The TOSCA (v170812) model covers the full CMS detector (18m × 120m domain, 8.7M mesh nodes). Post-processed into 11,136 primitive volumes for fast lookup + bilinear interpolation in CMSSW. Accuracy: **~0.1%** inside the tracker, ~3% in yoke steel.

The **missing turn** in coil module CB-2 (z = −3.77 m) shifts the axial field distribution ~15–20 mm toward +Z. TOSCA models this: OPERA predicts D > C by 3.37 mT vs. measured 3.6 mT — good agreement. The dominant NMR residual (~1.7–2.7 mT) is the overall scale, not the missing turn.

Key references:
- Amapane, Klyukhin. "Development of the CMS Magnetic Field Map." Symmetry 2023, 15, 1030. arXiv:2401.01913
- Klyukhin. "Design and Description of the CMS Magnetic System Model." Symmetry 2021, 13, 1052
- Maroussov PhD thesis, 2008 (local: `/work/submit/david_w/Documents/theses/maroussov.pdf`)
- Field-0045 (missing turn note, local: `/work/submit/david_w/Documents/notes/TeX-Field-0045.pdf`)

---

## Output NPZ File Format

Fit results are saved as `.npz` (numpy archive) with keys:

| Key | Description |
|---|---|
| `coeffs` | Fitted coefficients, shape `(n_params,)` |
| `params` | Parameter labels as strings `"l{l}_m{m}_{cs}"` |
| `l_max` | Maximum harmonic degree |
| `r_scale` | Normalisation radius [cm] |
| `z0` | Axial origin offset [cm] (0 unless --z0 was set) |
| `col_norms` | Column norms used for scaling (needed for eval) |
| `tikhonov` | Regularization strength λ |
| `tikhonov_power` | Spectral power s |
| `rms_Bz/Br/Bphi` | RMS residuals per component [T] |
| `condition_number` | Condition number of the scaled design matrix |
| `components` | Which components were fitted |
| `rmax_cm` | Cylindrical r cut applied during fitting [cm] |
| `zmax_cm` | \|z\| cut applied during fitting [cm] |
| `rmax_sphere_cm` | Spherical R cut applied during fitting [cm] |

The cuts (`rmax_cm`, `zmax_cm`, `rmax_sphere_cm`) are read back by `plot_residuals.py` to automatically reproduce the same data selection used during fitting.

---

## Naming Conventions for NPZ Output Files

Pattern: `{grid_stem}_coeffs_lmax{N}[_{tag}].npz`

- `_all3`: fit to Bz + Br + Bphi simultaneously
- `_noreg`: no Tikhonov regularization
- `_s{k}`: spectral Tikhonov with power s=k (and λ=1e-3 unless otherwise noted)
- `_rmax{R}`: non-default r cut (e.g. `_rmax308` tested going into the coil conductor)
- `_z0_{d}mm`: non-zero axial origin offset
- `_cyl{R}`: cylindrical (Fourier-Bessel) basis fit

Current best: `tosca170812_sphere320_coeffs_lmax18_noreg.npz` (lmax=18, no reg, tracker RMS 0.095 mT) or `tosca170812_sphere320_coeffs_lmax18_s3.npz` (lmax=18, λ=1e-3, s=3, tracker RMS 0.067 mT).
