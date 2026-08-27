# TIGRESS_SIM

Simulation and machine-learning tools for the **TIGRESS Gimpy B detector**. The repository uses **Geant4** to generate gamma-ray interaction positions and deposited energies, **SolidStateDetectors.jl (SSD)** to calculate electric fields, charge drift, and nine-contact waveforms, and **PyTorch** to reconstruct interaction coordinates from detector waveforms.

The main workflow is:

```text
        -> energy-deposition hits
        -> SSD charge-drift waveforms
        -> simulated and real superpulses
        -> simulation-to-real electronics fit
        -> event-by-event ML dataset
        -> PyTorch coordinate and uncertainty model
        -> coordinate predictions for real waveforms
```

## Repository components

### Geant4 simulations

The repository contains two Geant4 detector projects:

- **Realistic Gimpy B geometry**: octagonal mantle, two front taper cuts, and an offset rounded core bore.
- **Pure-cylinder geometry**: simplified cylindrical reference model without the octagonal sides or taper.

Both projects generate files such as:

```text
geant_hits.csv
geant_total_edep.csv
geant_gamma_tracks.csv
```

`geant_hits.csv` stores event IDs, hit coordinates in Geant4 and SSD conventions, and deposited energies. The full event energy must be summed before selecting 1332.5 keV photopeak events.

### SSD detector builders

- `build_gimpy_B_constant_thickness_rounded_core_offset_bore.jl` builds the realistic tapered and octagonal SSD model.
- `simple_geometry.jl` builds the simplified pure-cylinder SSD model.

The builders define the HPGe geometry, eight outer contacts, the core contact, bias voltage, temperature, and z-dependent impurity profile. They calculate the electric potential, electric field, and weighting potentials for contacts 1 through 9.

Principal outputs:

```text
gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls
gimpy_B_pure_cylinder_rounded_core_sim.jls
```


### Simulated superpulses

`ssd_superpulseV2.jl` converts Geant4 hits into SSD waveforms and averages clean 1332.5 keV events into one nine-contact superpulse for each outer collecting contact.

The selection includes:

- total Geant4 energy gate;
- single-contact Geant4 topology gate;
- Geant4 and SSD collecting-contact agreement;
- optional segment and taper boundary filters;
- SSD target and core amplitude gates;
- collecting-contact confidence;
- common core-trigger alignment.

### Real superpulses

`build_real_data_superpulses.jl` parses real nine-contact waveforms, subtracts channel baselines, converts ADC values to keV, selects clean 1332.5 keV events, and averages the accepted events by collecting contact.

The numeric field before `:` in the raw waveform file is ignored. All selection and energy calculations use the waveform samples.

### Simulation-to-real waveform fitting

`fitting_waveformsV2.jl` fits the simulated superpulses to the real superpulses. The fit estimates:

- target-specific global timing shifts;
- shared outer and core shaping times;
- core timing offset;
- residual channel delays;
- shared integral and differential cross talk matrices;
- regularized target-specific matrix corrections.

The fitted event-level transformation is:

```text
fitted = C_target * shaped_sim
       + D_target * derivative(shaped_sim)
```

Principal output:

```text
joint_sim_to_real_fit/hierarchical_sim_to_real_fit.jls
```

### ML waveform dataset

`fullysim_waveform.jl` creates the event-by-event dataset used by the PyTorch model.

The script:

- groups Geant4 hits by event;
- clusters nearby deposits;
- retains exactly one connected spatial cluster;
- applies a 1 mm detector-edge fiducial cut;
- calculates the energy-weighted cluster centroid;
- simulates all nine SSD contacts;
- applies the fitted timing, shaping, delay, `C`, and `D` transformations;
- creates deterministic contact-stratified train, validation, and test splits.

Main arrays:

```text
waveforms_keV   Float32 [N, 9, 100]
positions_mm    Float32 [N, 3]
```

Coordinate columns are:

```text
x_mm, y_mm, z_ssd_mm
```

Waveforms remain in keV and are not normalized event by event. Fixed standardization values are calculated from the training split only.

### Julia-to-PyTorch export

`julia-to-pytorch-data.jl` converts the serialized Julia dataset into NumPy-compatible binary arrays.

Julia is column-major, so the exporter explicitly permutes multidimensional arrays before writing them. Always verify the printed Julia and Python round-trip samples before training.

### PyTorch coordinate model

`CNN_train.py` trains the final two-stage one-dimensional CNN:

1. **Stage 1** trains the CNN backbone and coordinate head with MSE.
2. **Stage 2** reloads and freezes the best Stage 1 model, then trains a separate uncertainty head with Gaussian negative log likelihood.

The final model predicts:

```text
x, y, z
sigma_x, sigma_y, sigma_z
```

The Stage 2 checkpoint is self-contained and includes the frozen Stage 1 backbone and coordinate head.

### Real-waveform inference

`classifiy_real+waveforms.py` loads the final two-stage checkpoint and predicts coordinates from real nine-contact waveforms.

Outputs include:

```text
real_waveform_predictions.csv
real_hits_xy_uncertainty.png
real_hits_xy_uncertainty_clipped.png
real_uncertainty_histogram.png
real_predicted_z_histogram.png
```

The x-y hit map is colored using:

```text
sigma_3d_proxy = sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2)
```

The coordinate head is unconstrained and can produce nonphysical coordinates for out-of-distribution real waveforms. Such events should be flagged, not silently clipped to the detector boundary.

## Coordinate and geometry conventions

```text
Geant4 crystal center: z = 0 mm
Geant4 crystal extent: -45 to +45 mm
SSD crystal extent:     0 to 90 mm
```

Conversion:

```text
x_ssd = x_geant4
y_ssd = y_geant4
z_ssd = z_geant4 + 45 mm
```

Contact order:

```text
1, 2, 3, 4, 5, 6, 7, 8, core
```

## Quick start

### 1. Build the SSD detector

```bash
julia build_gimpy_B_constant_thickness_rounded_core_offset_bore.jl
```

### 2. Build and run the matching Geant4 project

```bash
source /data1/flerner/software/geant4-install/bin/geant4.sh
cd geant
mkdir -p build
cd build
cmake -DGeant4_DIR=/data1/flerner/software/geant4-install/lib/cmake/Geant4 ..
cmake --build . -j4
./ge_cylinder_test ../run.mac
```

### 3. Build simulated and real superpulses

```bash
julia ssd_superpulseV2.jl
julia build_real_data_superpulses.jl
```

### 4. Fit the simulated response to real data

```bash
julia fitting_waveformsV2.jl
```

### 5. Build the ML dataset

```bash
julia fullysim_waveform.jl
julia julia-to-pytorch-data.jl
```

### 6. Train and apply the ML model

```bash
python CNN_train.py
python 'classifiy_real+waveforms.py'
```

## Important notes

- The ML model is trained on simulated compact single-cluster events within a 1 mm fiducial edge margin.
- Predicted uncertainty estimates within-distribution residual uncertainty and does not by itself guarantee that a real waveform is in distribution.

Refer to the README files inside the Geant4, superpulse, fitting, and ML folders for more details.