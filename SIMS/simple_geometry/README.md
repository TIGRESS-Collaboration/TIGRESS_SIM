# Gimpy B Pure-Cylinder Geant4 and SSD Simulation

This folder contains the Geant4 particle-transport simulation and matching `SolidStateDetectors.jl` geometry builder.

The simplified detector is intentionally cylindrical. The model excludes the octagonal mantle and front taper so that geometry effects can be separated from electric-field, charge-drift, and electronics-response effects.

The two primary components are:

- the Geant4 project, which generates gamma-ray energy-deposition positions and energies;
- `simple_geometry.jl`, which builds the corresponding SSD detector, electric field, and nine weighting potentials.

The Geant4 hits are subsequently passed to SSD to calculate charge drift and produce nine-contact simulated waveforms and superpulses.

---

## Files and project components

### `simple_geometry.jl`

Builds the simplified pure-cylinder SSD detector.

The script generates:

```text
gimpy_B_pure_cylinder_rounded_core.yaml
gimpy_B_pure_cylinder_rounded_core_sim.jls
```

The serialized simulation contains:

- cylindrical HPGe geometry;
- centered rounded blind core bore;
- eight outer contacts and one core contact;
- linear impurity profile;
- electric potential;
- electric field;
- weighting potentials for contacts 1 through 9.

### Geant4 project

```text
gimpy_B_pure_cylinder_co60_geant4/
├── CMakeLists.txt
├── main.cc
├── run.mac
├── vis.mac
├── include/
│   ├── ActionInitialization.hh
│   ├── DetectorConstruction.hh
│   ├── EventAction.hh
│   ├── PrimaryGeneratorAction.hh
│   └── SteppingAction.hh
├── src/
│   ├── ActionInitialization.cc
│   ├── DetectorConstruction.cc
│   ├── EventAction.cc
│   ├── PrimaryGeneratorAction.cc
│   └── SteppingAction.cc
└── build/
```

The Geant4 project generates interaction truth for the same simplified cylindrical detector geometry.

---


# Detector geometry

The detector is a simplified version of the Gimpy B segmented HPGe detector.

## Crystal

- Material: high-purity germanium
- Geant4 material: `G4_Ge`
- SSD material: `HPGe`
- Shape: pure cylinder
- Outer radius: `30 mm`
- Height: `90 mm`
- Geant4 crystal center: `z = 0 mm`
- Geant4 crystal extent: `z = -45 mm` to `+45 mm`
- SSD crystal extent: `z = 0 mm` to `90 mm`
- Taper: not included
- Octagonal side shaping: not included

## Rounded blind core bore

- Bore radius: `5 mm`
- Core axis: centered at `x = 0 mm`, `y = 0 mm`
- Cylindrical section in SSD coordinates: `z = 0 mm` to `70 mm`
- Spherical-cap center in SSD coordinates: `z = 70 mm`
- Bore endpoint in SSD coordinates: `z = 75 mm`

The bore is formed from the union of:

```text
5-mm-radius cylinder
+
5-mm-radius sphere
```

This union is removed from the cylindrical germanium volume.

## Geant4-to-SSD coordinate conversion

Geant4 uses a crystal-centered z coordinate, while SSD uses a back-to-front coordinate.

```text
x_ssd = x_geant4
y_ssd = y_geant4
z_ssd = z_geant4 + 45 mm
```

Later SSD waveform scripts must use `z_ssd_mm` rather than `z_g4_mm`.

---

# SSD geometry builder: `simple_geometry.jl`

## Purpose

`simple_geometry.jl` creates the SSD geometry and performs the field calculations required for waveform simulation.

The simplified geometry removes:

```text
front taper cuts
octagonal side mask
offset core bore
```

The builder preserves:

```text
30 mm outer radius
90 mm crystal height
0.5 mm outer-contact shell
eight outer contacts
one core contact
rounded blind core bore
impurity profile
operating voltage
temperature
electric field
nine weighting potentials
```

## Run switches

```julia
const RUN_ELECTRIC_POTENTIAL = true
const RUN_ELECTRIC_FIELD = true
const RUN_WEIGHTING_POTENTIALS = true
```

All three should be enabled for final waveform production.

For quick geometry tests, weighting-potential generation may be disabled temporarily because it is the most expensive stage.

## Refinement limits

```julia
const REFINEMENT_LIMITS = [1.0, 0.5, 0.25]
```

Examples:

```julia
# Fast geometry test
const REFINEMENT_LIMITS = [1.0]

# Higher-resolution production field
const REFINEMENT_LIMITS = [1.0, 0.5, 0.25, 0.1]
```

Smaller refinement limits increase spatial resolution, runtime, and memory use.

## Detector parameters

```julia
const CRYSTAL_NAME = "Gimpy_B_Pure_Cylinder"
const SERIAL = 74037

const VD = 2500.0
const VOP = 3500.0
const TEMPERATURE = 103.0
```

The eight outer contacts are held at `0 V`. The core is held at `3500 V`.

## Impurity profile

```julia
const IMP_BOTTOM = 0.95
const IMP_TOP = 2.25
```

These values are in units of:

```text
1e10 cm^-3
```

The generated YAML defines the impurity gradient along SSD z only:

```yaml
impurity_density:
  name: linear
  offset: 9.5e9cm^-3
  gradient:
    z: 1.4444444444444444e9cm^-4
```

Therefore:

```text
SSD z = 0 mm:  0.95e10 cm^-3
SSD z = 90 mm: 2.25e10 cm^-3
```


## Cylindrical geometry parameters

```julia
const R_OUTER = 30.0
const CRYSTAL_HEIGHT = 90.0
const CRYSTAL_CENTER_Z = CRYSTAL_HEIGHT / 2
```

The SSD cylinder is centered at:

```text
z = 45 mm
```

and spans:

```text
z = 0 mm to 90 mm
```

## Constant-thickness outer-contact shell

```julia
const OUTER_CONTACT_THICKNESS = 0.5
const R_INNER_CONTACT = R_OUTER - OUTER_CONTACT_THICKNESS
```

This gives:

```text
outer radius:       30.0 mm
inner shell radius: 29.5 mm
shell thickness:     0.5 mm
```

Each outer contact is formed by intersecting:

```text
its phi/z segment selector
```

with:

```text
outer cylindrical crystal - inner cylindrical crystal
```

The result is a constant 0.5 mm radial shell on the outer mantle.

## Segment selector radius

```julia
const SEGMENT_SELECTOR_RADIUS = R_OUTER + 1.0
```

The selector radius is 31 mm. This value does not define the physical detector radius. It is intentionally oversized so the selector only controls the phi and z ranges.

## Contact segmentation

### Front contacts

Front contacts cover `z = 70 mm` to `90 mm`:

```text
Contact 1: phi  90 to 180 degrees
Contact 2: phi 180 to 270 degrees
Contact 3: phi 270 to 360 degrees
Contact 4: phi   0 to  90 degrees
```

### Back contacts

Back contacts cover `z = 0 mm` to `70 mm`:

```text
Contact 5: phi  90 to 180 degrees
Contact 6: phi 180 to 270 degrees
Contact 7: phi 270 to 360 degrees
Contact 8: phi   0 to  90 degrees
```

### Core

```text
Contact 9: centered rounded core bore
Potential: VOP = 3500 V
```

## SSD grid

The generated YAML uses:

```text
x: -35 mm to +35 mm
y: -35 mm to +35 mm
z:  -5 mm to +95 mm
```

The grid extends beyond the active semiconductor into surrounding vacuum.

## YAML generation

The script writes:

```text
gimpy_B_pure_cylinder_rounded_core.yaml
```

The YAML contains:

- units;
- Cartesian grid;
- vacuum medium;
- HPGe semiconductor;
- temperature;
- z-dependent linear impurity density;
- cylindrical geometry;
- rounded bore;
- all nine contact definitions and potentials.

The first 100 lines are printed for inspection.

## Contact validation

After loading the YAML, the script verifies that the detector contains exactly contacts 1 through 9:

```julia
contact_ids = sort([contact.id for contact in sim.detector.contacts])
expected_contact_ids = collect(1:9)
```

The script stops if the contact list is incorrect.

## Electric potential and field

```julia
calculate_electric_potential!(
    sim,
    refinement_limits = REFINEMENT_LIMITS
)
```

```julia
calculate_electric_field!(sim)
```

## Weighting potentials

```julia
const CONTACTS_FOR_WEIGHTING = 1:9
```

```julia
calculate_weighting_potential!(
    sim,
    contact_id,
    refinement_limits = REFINEMENT_LIMITS
)
```

All nine weighting potentials are required for complete waveform simulation.

## Crystal orientation

`simple_geometry.jl` calculates geometry, electric fields, and weighting potentials, but it does not assign the anisotropic ADL charge-drift orientation.

The orientation is attached later during waveform simulation, for example:

```julia
const PHI110_DEG = 45.0

charge_drift_model = ADLChargeDriftModel(
    T = T,
    temperature = 103.0,
    phi110 = PHI110_DEG
)

sim.detector = SolidStateDetector(
    sim.detector,
    charge_drift_model
)
```

With the current `phi110` convention:

```text
crystal [001] is aligned with detector z
crystal [110] lies in the detector x-y plane
phi110 is measured from detector +x to crystal [110]
```

For `phi110 = 45 degrees`:

```text
[110] direction = (0.7071, 0.7071, 0)
```

This does not mean that the detector z-axis is `[110]`.

## Serialized SSD output

The completed simulation is saved as:

```text
gimpy_B_pure_cylinder_rounded_core_sim.jls
```

Later scripts load it with:

```julia
using Serialization

sim = deserialize(
    "gimpy_B_pure_cylinder_rounded_core_sim.jls"
)
```

## Running the SSD builder

```bash
julia simple_geometry.jl
```

Or from Julia:

```julia
include("simple_geometry.jl")
```

The output files are written beside the script because:

```julia
const OUTDIR = @__DIR__
```

---

# Geant4 source configuration

The source is defined in `src/PrimaryGeneratorAction.cc` using `G4ParticleGun`.

## Gamma energy

The default superpulse-production configuration emits one monoenergetic gamma per event:

```text
Gamma energy: 1332.5 keV
```

The source code contains the principal Co-60 gamma energies:

```cpp
const G4double CO60_E1 = 1173.2 * keV;
const G4double CO60_E2 = 1332.5 * keV;
```

For clean superpulse production:

```cpp
const bool EMIT_BOTH_CO60_GAMMAS = false;
```

Setting the flag to `true` emits both gamma rays in every event.

## Source position

```cpp
const G4double SOURCE_X = 0.0 * mm;
const G4double SOURCE_Y = 0.0 * mm;
const G4double SOURCE_Z = 145.0 * mm;
```

The Geant4 front face is at `z = +45 mm`, so the source is 100 mm in front of the detector.

## Direction cone

Gamma directions are sampled uniformly in solid angle inside a cone pointing toward global negative z.

```cpp
const G4double CONE_HALF_ANGLE = 20.0 * deg;
```

A radius-30 mm face at 100 mm distance subtends approximately 16.7 degrees, so 20 degrees provides a small margin.

The axial component must point toward the detector:

```cpp
G4double uz = -cosTheta;
```

---

# Physics configuration

Recommended physics list:

```cpp
new FTFP_BERT()
```

The compact-output setup does not manually enable:

```text
fluorescence
Auger electron production
PIXE
```

Explicit activation generated extremely large numbers of microscopic deposits in testing.

Germanium impurity is not represented in Geant4. The impurity profile affects electric fields and charge drift rather than gamma interaction probability at HPGe impurity concentrations.

---

# Building the Geant4 simulation

Load the Geant4 environment:

```bash
source /data1/flerner/software/geant4-install/bin/geant4.sh
```

Configure and compile:

```bash
cd /data1/flerner/hpge_sims/gimpy_B_pure_cylinder_co60_geant4

rm -rf build
mkdir build
cd build

cmake \
  -DGeant4_DIR=/data1/flerner/software/geant4-install/lib/cmake/Geant4 \
  ..

cmake --build . -j4
```

The expected executable is:

```text
gimpy_b_co60_cone
```

Confirm the actual target with:

```bash
grep -n "add_executable" ../CMakeLists.txt
```

---

# Running the Geant4 simulation

From the build directory:

```bash
source /data1/flerner/software/geant4-install/bin/geant4.sh
./gimpy_b_co60_cone ../run.mac
```

Production event count example:

```text
/run/beamOn 2000000
```

Use a smaller test first:

```text
/run/beamOn 10000
```

Source position, gamma energy, and cone angle are defined in `PrimaryGeneratorAction.cc`, not necessarily in `run.mac`.

---

# Geant4 output files

Outputs are written in the directory from which the executable is run, normally `build`.

## `geant_hits.csv`

```text
event_id,track_id,x_mm,y_mm,z_g4_mm,z_ssd_mm,edep_keV
```

Each row is one positive energy-deposition step.

The hit position is the midpoint between pre-step and post-step positions. Midpoints reduce the chance that SSD later interprets a boundary point as outside the semiconductor.

## `geant_total_edep.csv` or `geant_event_totals.csv`

```text
event_id,total_edep_keV
```

Use the complete event total to identify 1332.5 keV full-energy events.

## `geant_gamma_tracks.csv`

Contains gamma-track and step diagnostics. This file is optional for waveform production and may become large.

---

# Creating SSD waveforms and superpulses

Filtered Geant4 hits are passed to:

```text
pure_cylinder_superpulses.jl
```

The waveform script should load:

```text
gimpy_B_pure_cylinder_rounded_core_sim.jls
```


A later per-hit threshold may reduce tiny SSD charge clouds:

```julia
const MIN_HIT_EDEP_KEV = 5.0
```

Do not apply the per-hit threshold before computing the full-event energy.

## SSD position conversion

Use:

```julia
CartesianPoint{Float32}(
    Float32(x_mm * 1e-3),
    Float32(y_mm * 1e-3),
    Float32(z_ssd_mm * 1e-3)
)
```

Convert millimetres to metres exactly once.

If the simulation is `Simulation{Float32}`, event positions and relevant SSD parameters must also use `Float32`.

---

# Important limitations

The simplified model intentionally omits:

```text
octagonal mantle
front taper cuts
offset core bore
```

The Geant4 model may also omit:

```text
cryostat
vacuum gap
dead layers
electrode materials
mounting structures
```

These can affect efficiency, continuum shape, interaction-depth distribution, and surface-event populations.

The pure-cylinder model is primarily a geometry-control and waveform-development tool. The realistic tapered and octagonal model should be used for final Gimpy B production unless the cylindrical approximation is specifically intended.

---


# Quick start

## Build the SSD detector

```bash
julia simple_geometry.jl
```

## Build and run Geant4

```bash
source /data1/flerner/software/geant4-install/bin/geant4.sh
cd /data1/flerner/hpge_sims/gimpy_B_pure_cylinder_co60_geant4
rm -rf build
mkdir build
cd build
cmake -DGeant4_DIR=/data1/flerner/software/geant4-install/lib/cmake/Geant4 ..
cmake --build . -j4
./gimpy_b_co60_cone ../run.mac
```

Principal SSD output:

```text
gimpy_B_pure_cylinder_rounded_core_sim.jls
```

Principal Geant4 output:

```text
build/geant_hits.csv
```