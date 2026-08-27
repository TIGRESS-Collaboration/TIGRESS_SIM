# Pure-cylinder Co-60-style cone setup

This version matches the supplied shaped-detector source and stepping setup:

- `G4ParticleGun`, not GPS
- off-corner source position aimed at contact 3
- cone centered on global `-z`
- 40-degree half-angle
- 1332.5-keV gamma only by default
- optional 1173.2-keV second gamma controlled by `EMIT_BOTH_CO60_GAMMAS`
- gamma-track CSV plus hit CSV
- event-total energy CSV

The active-volume check uses `GimpyBPureCylinderActiveCrystal`, matching the
pure-cylinder detector construction. Output files are:

- `geant_hits.csv`
- `geant_gamma_tracks.csv`
- `geant_event_totals.csv`

Build:

```bash
source /data1/flerner/software/geant4-install/bin/geant4.sh
mkdir -p build && cd build
cmake -DGeant4_DIR=/data1/flerner/software/geant4-install/lib/cmake/Geant4 ..
cmake --build . -j4
./gimpy_b_co60_cone ../run.mac
```
