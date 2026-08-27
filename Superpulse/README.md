# Gimpy B HPGe Superpulse Fitting and ML Dataset Pipeline

This folder contains the Julia scripts used to generate simulated and measured Gimpy B HPGe superpulses, fit the simulated detector response to measured electronics behavior, and produce the final event-by-event dataset used for machine-learning position reconstruction.


The scripts should normally be run in this order:

```bash
julia ssd_superpulseV2.jl
julia build_real_data_superpulses.jl
julia fitting_waveformsV2.jl
julia fullysim_waveform.jl
```
---

## Files

### `ssd_superpulseV2.jl`

Builds contact-specific simulated superpulses from Geant4 hit positions and deposited energies.

The script:

- loads the serialized Gimpy B SSD simulation;
- attaches the ADL charge-drift model;
- reads Geant4 hits grouped by event;
- applies a 1332.5 keV photopeak gate;
- classifies Geant4 energy by expected outer contact;
- rejects Geant4 events that deposit substantial energy in multiple contacts;
- optionally rejects events near segment and taper boundaries;
- runs SSD charge drift with diffusion and self-repulsion;
- requires the Geant4 and SSD collecting contacts to agree;
- applies target, core-energy, and collecting-contact confidence gates;
- aligns accepted events using the core constant-fraction trigger;
- averages accepted events into one nine-contact superpulse per collecting contact;
- writes serialized outputs, plots, and detailed diagnostics.

### `build_real_data_superpulses.jl`

Builds contact-specific measured superpulses from real event waveforms.

The script:

- parses complete nine-contact events from `out50k.dat`;
- ignores the numeric field before `:` on each waveform line;
- truncates or pads records to 100 samples;
- baseline-subtracts each channel;
- converts every contact from ADC units to keV;
- identifies the collecting outer contact from final waveform levels;
- applies a 1332.5 keV waveform gate;
- applies a strict final-charge confidence cut;
- averages accepted events into measured superpulses;
- optionally saves selected individual waveforms;
- writes serialized outputs, plots, parser diagnostics, and event summaries.

### `fitting_waveformsV2.jl`

Fits the simulated contact-specific superpulses to the measured superpulses.

The fit estimates:

- target-specific global timing shifts;
- shared outer-contact shaping;
- shared core shaping;
- a shared core-versus-outer delay;
- residual per-channel delays;
- shared integral and differential electronics matrices;
- regularized target-specific matrix corrections.

The fitted parameters are used by `fullysim_waveform.jl` to transform individual SSD events into waveforms that better resemble measured data.

### `fullysim_waveform.jl`

Builds the final event-by-event ML dataset.

The script:

- reads and groups Geant4 hits;
- removes tiny deposits;
- forms spatial hit clusters;
- retains events containing exactly one connected cluster;
- applies a 1 mm detector-edge fiducial cut;
- calculates the energy-weighted cluster centroid;
- simulates all nine SSD waveforms;
- applies the fitted gain, timing, shaping, delay, `C`, and `D` transformations;
- stores contiguous `Float32` waveform and label arrays;
- creates deterministic contact-stratified train, validation, and test splits.

---

# Detector conventions

## Contacts

```text
Contacts 1 through 8: outer segmented contacts
Contact 9: core, labeled FV in real waveform files
```

Waveform matrices always use:

```text
row 1 -> contact 1
row 2 -> contact 2
...
row 8 -> contact 8
row 9 -> core
```

## Segmentation

```text
Front contacts, z >= 70 mm:
  Contact 1:  90 <= phi < 180 degrees
  Contact 2: 180 <= phi < 270 degrees
  Contact 3: 270 <= phi < 360 degrees
  Contact 4:   0 <= phi <  90 degrees

Back contacts, z < 70 mm:
  Contact 5:  90 <= phi < 180 degrees
  Contact 6: 180 <= phi < 270 degrees
  Contact 7: 270 <= phi < 360 degrees
  Contact 8:   0 <= phi <  90 degrees
```

## Coordinates

All final labels use SSD Cartesian coordinates in millimetres:

```text
x_mm
y_mm
z_ssd_mm
```

Use `z_ssd_mm` rather than an unshifted Geant4 z coordinate when passing hits to SSD or creating ML labels.

---

# Requirements

The scripts use these Julia packages:

```julia
using SolidStateDetectors
using Serialization
using Unitful
using Statistics
using LinearAlgebra
using Random
using Printf
using TypedTables
using Plots
```

Install missing external packages in the active Julia environment:

```julia
import Pkg
Pkg.add("SolidStateDetectors")
Pkg.add("Unitful")
Pkg.add("TypedTables")
Pkg.add("Plots")
```

For headless plotting, use:

```julia
ENV["GKSwstype"] = "100"
```

---

# 1. Simulated superpulse generation

## Inputs

```julia
const SIM_FILE = "/data1/flerner/hpge_sims/CNN/gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls"
const GEANT_HITS_FILE = "/data1/flerner/hpge_sims/geant/build/geant_hits.csv"
```

The Geant4 CSV must contain:

```text
event_id
track_id
x_mm
y_mm
z_ssd_mm
edep_keV
```

## ADL and SSD settings

```julia
const PHI110_DEG = 45.0
const TEMPERATURE_K = 103.0
const DRIFT_DT = 2u"ns"
const MAX_NSTEPS = 20_000
const RUN_DIFFUSION = true
const RUN_SELF_REPULSION = true
```

## Simulated output grid

```julia
const ML_DT_NS = 2.0
const ML_TRACE_LENGTH_NS = 1200.0
```

This produces 601 samples from 0 to 1200 ns.

## Event selection

The Geant4 photopeak gate is:

```julia
const PHOTOPEAK_ENERGY_KEV = 1332.5
const PHOTOPEAK_TOLERANCE_KEV = 10.0
```

The Geant4 single-contact gate requires:

```julia
const MIN_GEANT_CONTACT_FRACTION = 0.90
const MAX_GEANT_SECOND_CONTACT_ENERGY_KEV = 60.0
```

The collecting contact inferred from the SSD waveform must match the Geant4 contact classification:

```julia
const REQUIRE_GEANT_SSD_CONTACT_AGREEMENT = true
```

The SSD collecting-contact confidence threshold is:

```julia
const MIN_SEGMENT_CONFIDENCE = 0.90
```

## Boundary filtering

The segment-boundary filter is optional:

```julia
const APPLY_SEGMENT_BOUNDARY_FILTER = false
const PHI_BOUNDARY_MARGIN_DEG = 3.0
const Z_BOUNDARY_MARGIN_MM = 2.0
```

The taper-boundary filter is enabled:

```julia
const APPLY_TAPER_BOUNDARY_FILTER = true
const TAPER_BOUNDARY_MARGIN_MM = 2
```

## Trigger alignment

```julia
const APPLY_TRIGGER_ALIGNMENT = true
const TRIGGER_CONTACT_ID = CORE_CONTACT_ID
const TRIGGER_FRACTION = 0.10
const TRIGGER_TIME_NS = 300.0
```

All nine contacts receive the same event-level shift.

## Optional electronics

The pristine simulated superpulse configuration keeps these disabled:

```julia
const APPLY_PREAMP_RESPONSE = false
const APPLY_INTEGRAL_CROSSTALK = false
const APPLY_DIFFERENTIAL_CROSSTALK = false
const ADD_NOISE = false
```

Keep these disabled when electronics behavior will be fitted later. Otherwise, shaping or coupling may be applied twice.

## Outputs

Default directory:

```text
/data1/flerner/hpge_sims/CNN/gimpytests
```

Main output:

```text
repulsion_gimpy_geant_ssd_superpulses.jls
```

Important additional outputs:

```text
repulsion_gimpy_geant_ssd_superpulses_contact1.jls
...
repulsion_gimpy_geant_ssd_superpulses_contact8.jls
repulsion_gimpy_geant_ssd_superpulses_summary.csv
repulsion_gimpy_geant_ssd_superpulses_amplitude_summary.csv
repulsion_gimpy_geant_ssd_superpulses_accepted_positions.csv
repulsion_gimpy_geant_ssd_superpulses_contact_diagnostics.csv
repulsion_gimpy_geant_ssd_superpulses_ssd_failure_diagnostics.csv
```

Run:

```bash
julia ssd_superpulseV2.jl
```

---

# 2. Real superpulse generation

## Input format

```julia
const IN_WAVEFORM_FILE = "/data1/flerner/hpge_sims/CNN/realdata/out50k.dat"
```

Expected waveform-line format:

```text
contact_label, numeric_field : sample_1, sample_2, ..., sample_100
```

The numeric field before `:` is ignored. All selection and calibration decisions come from waveform samples.

## Real waveform grid

```julia
const NSAMPLES = 100
const SAMPLE_PERIOD_NS = 10.0
```

The time axis spans 0 to 990 ns.

## Baseline and final windows

```julia
const BASELINE_START_SAMPLE = 1
const BASELINE_END_SAMPLE = 30
const FINAL_START_SAMPLE = 90
const FINAL_END_SAMPLE = 99
```

## ADC-to-keV calibration

The active `WAVEFORM_1332_ADC` dictionary stores the measured 1332.5 keV ADC amplitude for every contact.

The conversion is:

```text
contact ADC-to-keV factor = 1332.5 / contact 1332 ADC amplitude
```

The older quoted calibration dictionary is inactive documentation.

## Event selection

```julia
const APPLY_1332_WAVEFORM_GATE = true
const ENERGY_GATE_HALF_WIDTH_KEV = 10.0
const APPLY_CONFIDENCE_CUT = true
const MIN_FINAL_CONFIDENCE = 0.98
```

The target is the outer contact with the largest absolute final level. The event energy is the magnitude of that final level.

## Outputs

Default directory:

```text
/data1/flerner/hpge_sims/CNN/real_waveform_only_superpulses
```

Main output:

```text
real_waveform_only_superpulses.jls
```

Important additional outputs:

```text
real_waveform_only_superpulses_contact1.jls
...
real_waveform_only_superpulses_contact8.jls
real_waveform_only_superpulses_event_summary.csv
real_waveform_only_superpulses_amplitude_summary.csv
real_waveform_only_superpulses_parser_summary.csv
```

Run:

```bash
julia build_real_data_superpulses.jl
```

---

# 3. Simulation-to-real electronics fit

## Inputs

```julia
const REAL_FILE = "/data1/flerner/hpge_sims/CNN/real_waveform_only_superpulses/real_waveform_only_superpulses.jls"
const SIM_FILE = "/data1/flerner/hpge_sims/CNN/gimpytests/repulsion_gimpy_geant_ssd_superpulses.jls"
```

## Baseline subtraction and core polarity

```julia
const REAL_BASELINE_END_NS = 290.0
const SIM_BASELINE_END_NS = 50.0
const FLIP_SIM_CORE_POLARITY = true
```

## Alignment

Initial target alignment uses derivative cross-correlation of the core signal, followed by target-specific refinement.

```julia
const XC_ORIGINAL_MIN_SHIFT_NS = -400.0
const XC_ORIGINAL_MAX_SHIFT_NS = 400.0
const REFINE_SHIFT_HALF_WINDOW_NS = 50.0
const REFINE_SHIFT_STEP_NS = 0.5
```

## Fit windows

```julia
const FIT_START_NS = 250.0
const FIT_END_NS = 950.0
const EDGE_START_NS = 300.0
const EDGE_END_NS = 500.0
```

## Shared shaping

The fit determines:

```text
outer_tau_ns
core_tau_ns
core_delay_ns
```

Both outer and core shaping use a two-pole low-pass response.

## Residual channel delays

```julia
const FIT_CHANNEL_DELAYS = true
const CHANNEL_DELAY_MIN_NS = -20.0
const CHANNEL_DELAY_MAX_NS = 20.0
const CHANNEL_DELAY_FINE_STEP_NS = 0.25
const FIX_CORE_RESIDUAL_DELAY = true
```

The shared core delay already represents the core-versus-outer timing difference, so the residual core-channel delay is fixed at zero.

## Hierarchical crosstalk

The event-level fitted response is:

```text
fitted = C_target * shaped_sim
       + D_target * derivative(shaped_sim)
```

The fitter first estimates shared electronics matrices:

```text
C_shared
D_shared
```

It then estimates regularized target-specific corrections:

```text
C_target = C_shared + deltaC_target
D_target = D_shared + deltaD_target
```

The integral coefficients are dimensionless. Differential coefficients are stored in nanoseconds.

## Robust regression

The matrix fit uses ridge regression with Huber iteratively reweighted least squares. Collecting and core identity priors are used, but mirror channels are not forced to identity behavior.

## Outputs

Default directory:

```text
/data1/flerner/hpge_sims/CNN/joint_sim_to_real_fit
```

Main output:

```text
hierarchical_sim_to_real_fit.jls
```

Important serialized keys:

```text
real
sim_raw_aligned
sim_shaped
sim_fitted
residual
shared_integral_crosstalk
shared_differential_crosstalk_ns
target_integral_crosstalk
target_differential_crosstalk_ns
target_integral_corrections
target_differential_corrections_ns
global_shifts_ns
channel_delays_ns
outer_tau_ns
core_tau_ns
core_delay_ns
time_ns
metadata
```

Important additional outputs:

```text
hierarchical_sim_to_real_fit_metrics.csv
hierarchical_sim_to_real_fit_shaping.csv
hierarchical_sim_to_real_fit_channel_delays.csv
hierarchical_sim_to_real_fit_shared_integral_crosstalk.csv
hierarchical_sim_to_real_fit_shared_differential_crosstalk_ns.csv
hierarchical_sim_to_real_fit_target_integral_crosstalk.csv
hierarchical_sim_to_real_fit_target_differential_crosstalk_ns.csv
hierarchical_sim_to_real_fit_target1.png
...
hierarchical_sim_to_real_fit_target8_residuals.png
```

Run:

```bash
julia fitting_waveformsV2.jl
```

---

# 4. ML waveform dataset generation

## Inputs

```julia
const GEANT_HITS_FILE = "/data1/flerner/hpge_sims/geant/build/geant_hits.csv"
const SSD_SIM_FILE = "/data1/flerner/hpge_sims/CNN/gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls"
const FIT_FILE = "/data1/flerner/hpge_sims/CNN/joint_sim_to_real_fit/hierarchical_sim_to_real_fit.jls"
```

The Geant4 CSV must contain:

```text
event_id
track_id
x_mm
y_mm
z_g4_mm
z_ssd_mm
edep_keV
```

## Single-cluster selection

```julia
const HIT_CLUSTER_RADIUS_MM = 1.0
const MIN_HIT_ENERGY_KEV = 0.001
const REQUIRE_EXACTLY_ONE_CLUSTER = true
const MAX_SPATIAL_EXTENT_MM = Inf
```

Hits are connected when their Euclidean separation is at most 1 mm. Connectedness is transitive.

The coordinate label is the energy-weighted centroid:

```text
centroid = sum(hit energy * hit position) / total deposited energy
```

## Detector-edge cut

```julia
const MIN_DETECTOR_EDGE_DISTANCE_MM = 1.0
const REQUIRE_ALL_HITS_INSIDE_EDGE_MARGIN = true
```

The centroid and every retained hit must be at least 1 mm from the nearest modeled detector boundary.

The edge model includes:

- outer cylinder;
- octagonal flats;
- top and bottom faces;
- two tapered top planes;
- offset cylindrical core bore;
- rounded blind bore cap.

Stored edge values are:

```text
centroid_edge_distance_mm
minimum_hit_edge_distance_mm
```

The edge cut defines a fiducial interior dataset. Report model accuracy together with this acceptance requirement.

## Energy range

```julia
const MIN_TOTAL_EDEP_KEV = 0.0
const MAX_TOTAL_EDEP_KEV = Inf
```

All total energies are accepted by default. Waveform amplitudes remain in keV.

Do not divide each event by its own energy or maximum amplitude.

## SSD and output grids

```julia
const SSD_SAMPLE_PERIOD_NS = 1.0
const SSD_MAX_NSTEPS = 2000
const OUTPUT_SAMPLE_PERIOD_NS = 10.0
const OUTPUT_NSAMPLES = 100
```

The final stored waveform shape is:

```text
N events x 9 contacts x 100 samples
```

The time grid spans 0 to 990 ns.

## Electronics transformation

For each accepted event, the script:

1. simulates the native SSD waveform;
2. identifies the collecting outer contact;
3. applies the fitted core-polarity convention;
4. applies the target amplitude gain when available;
5. applies the fitted target shift;
6. applies outer or core two-pole shaping;
7. applies the shared core delay;
8. applies residual channel delays;
9. resamples to the 10 ns output grid;
10. computes the derivative matrix;
11. applies target-specific `C` and `D` matrices.

```text
corrected = C_target * processed
          + D_target * derivative(processed)
```

## Data splits

```julia
const TRAIN_FRACTION = 0.70
const VALIDATION_FRACTION = 0.15
const TEST_FRACTION = 0.15
const SPLIT_RANDOM_SEED = 74037
const STRATIFY_SPLITS_BY_CONTACT = true
```

Splits are deterministic and approximately stratified by collecting contact.

## Main arrays

```text
waveforms_keV                 Float32 [N, 9, 100]
positions_mm                  Float32 [N, 3]
hit_sigma_xyz_mm              Float32 [N, 3]
total_energy_keV              Float32 [N]
spatial_extent_mm             Float32 [N]
max_cluster_radius_mm         Float32 [N]
centroid_edge_distance_mm     Float32 [N]
minimum_hit_edge_distance_mm  Float32 [N]
collecting_contact            Int8    [N]
event_id                      Int64   [N]
raw_hit_count                 Int16   [N]
time_ns                       Float32 [100]
train_indices
validation_indices
test_indices
train_channel_mean_keV
train_channel_std_keV
```

The coordinate columns are:

```text
x_mm, y_mm, z_ssd_mm
```

## Standardization

Waveforms are stored in keV without eventwise normalization.

Optional training standardization uses fixed statistics calculated only from the training split:

```text
standardized waveform =
    (waveform - training channel mean) /
    training channel standard deviation
```

The same stored values must be used for validation, testing, and real inference.

## Outputs

Default directory:

```text
/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset
```

Main output:

```text
gimpyB_single_cluster_waveforms.jls
```

Additional outputs:

```text
gimpyB_single_cluster_waveforms_manifest.csv
gimpyB_single_cluster_waveforms_README.txt
```

Run:

```bash
julia fullysim_waveform.jl
```

---

# Important implementation checks

## Geant4-to-SSD precision

The detector simulation uses `Float32`. Event positions, energies, times, sample interval, and interaction distance must therefore use `Float32`.

Expected position type:

```text
CartesianPoint{Float32}
```

Mixing `CartesianPoint{Float64}` with `Simulation{Float32}` causes a method error inside SSD charge clustering.

## CSV line numbering

Julia supports:

```julia
enumerate(eachline(io))
```

For physical line numbers after consuming a header:

```julia
for (line_offset, line) in enumerate(eachline(io))
    line_number = line_offset + 1
end
```

Julia does not support Python-style `enumerate(iterator, 2)`.

## Julia-to-Python array order

Julia arrays are column-major. NumPy and PyTorch normally use C-order.

Any raw-binary exporter must explicitly permute:

```text
N x 9 x 100  ->  100 x 9 x N
```

before writing, allowing NumPy to reconstruct C-order `N x 9 x 100`.

Always compare reference waveform samples and coordinate labels between Julia and Python before training.

---

# Suggested folder structure

```text
.
├── README.md
├── ssd_superpulseV2.jl
├── build_real_data_superpulses.jl
├── fitting_waveformsV2.jl
├── fullysim_waveform.jl
├── gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls
├── gimpytests/
│   ├── repulsion_gimpy_geant_ssd_superpulses.jls
│   ├── repulsion_gimpy_geant_ssd_superpulses_contact1.jls
│   ├── ...
│   ├── repulsion_gimpy_geant_ssd_superpulses_summary.csv
│   └── repulsion_gimpy_geant_ssd_superpulses_contact_diagnostics.csv
├── real_waveform_only_superpulses/
│   ├── real_waveform_only_superpulses.jls
│   ├── real_waveform_only_superpulses_contact1.jls
│   ├── ...
│   ├── real_waveform_only_superpulses_event_summary.csv
│   └── real_waveform_only_superpulses_parser_summary.csv
├── joint_sim_to_real_fit/
│   ├── hierarchical_sim_to_real_fit.jls
│   ├── hierarchical_sim_to_real_fit_metrics.csv
│   ├── hierarchical_sim_to_real_fit_shaping.csv
│   ├── hierarchical_sim_to_real_fit_channel_delays.csv
│   ├── hierarchical_sim_to_real_fit_target_integral_crosstalk.csv
│   └── hierarchical_sim_to_real_fit_target_differential_crosstalk_ns.csv
└── ml_single_cluster_dataset/
    ├── gimpyB_single_cluster_waveforms.jls
    ├── gimpyB_single_cluster_waveforms_manifest.csv
    └── gimpyB_single_cluster_waveforms_README.txt
```

---

# Full quick start

```bash
julia ssd_superpulseV2.jl
julia build_real_data_superpulses.jl
julia fitting_waveformsV2.jl
julia fullysim_waveform.jl
```

The final product of this folder is:

```text
ml_single_cluster_dataset/gimpyB_single_cluster_waveforms.jls
```

This dataset can then be exported to NumPy-compatible files and used to train the PyTorch waveform-to-coordinate model.