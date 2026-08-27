# ============================================================
# simple_geometry.jl
#
# Simplified Gimpy B segmented HPGe detector for
# SolidStateDetectors.jl.
#
# Differences from the original geometry:
#   - no taper cuts
#   - no octagonal side mask
#   - pure cylindrical outer crystal
#
# Preserved features:
#   - 30 mm outer radius
#   - 90 mm crystal height
#   - 0.5 mm constant-thickness outer contact shell
#   - eight outer contacts with the same phi/z segmentation
#   - rounded blind-end core bore
#   - Mikey A/Gimpy B impurity gradient
#   - electric potential and electric field calculation
#   - optional weighting-potential calculation for contacts 1:9
#
# Outputs:
#   gimpy_B_pure_cylinder_rounded_core.yaml
#   gimpy_B_pure_cylinder_rounded_core_sim.jls
# ============================================================

using SolidStateDetectors
using Serialization

# ============================================================
# Run switches
# ============================================================

const RUN_ELECTRIC_POTENTIAL = true
const RUN_ELECTRIC_FIELD = true
const RUN_WEIGHTING_POTENTIALS = true

# For fast geometry tests, use [1.0].
# For the higher-resolution field calculation, use:
# [1.0, 0.5, 0.25, 0.1]
const REFINEMENT_LIMITS = [1.0, 0.5, 0.25]

# Calculate all weighting potentials for waveform simulation.
const CONTACTS_FOR_WEIGHTING = 1:9

# ============================================================
# Detector parameters
# ============================================================

const CRYSTAL_NAME = "Gimpy_B_Pure_Cylinder"
const SERIAL = 74037

const VD = 2500.0
const VOP = 3500.0
const TEMPERATURE = 103.0

# Impurity values in units of 1e10 cm^-3.
# z = 0 mm is the more-pure end.
# z = 90 mm is the less-pure end.
const IMP_BOTTOM = 0.95
const IMP_TOP = 2.25

const IMP_OFFSET_CM3 = IMP_BOTTOM * 1e10
const IMP_GRADIENT_CM4 =
    (IMP_TOP - IMP_BOTTOM) * 1e10 / 9.0

# ============================================================
# Cylindrical geometry parameters in mm
# ============================================================

const R_OUTER = 30.0
const CRYSTAL_HEIGHT = 90.0
const CRYSTAL_CENTER_Z = CRYSTAL_HEIGHT / 2

# Constant radial thickness of the eight outer contacts.
const OUTER_CONTACT_THICKNESS = 0.5
const R_INNER_CONTACT =
    R_OUTER - OUTER_CONTACT_THICKNESS

# Rounded blind-end core bore.
# The cylindrical section extends from z = 0 to z = 70 mm.
# The sphere is centered at z = 70 mm and reaches z = 75 mm.
const CORE_RADIUS = 5.0
const CORE_HEIGHT = 75.0
const CORE_CAP_RADIUS = CORE_RADIUS
const CORE_CYL_HEIGHT =
    CORE_HEIGHT - CORE_CAP_RADIUS
const CORE_CYL_CENTER_Z =
    CORE_CYL_HEIGHT / 2
const CORE_CAP_CENTER_Z =
    CORE_CYL_HEIGHT

# Segment selector radius. This is intentionally slightly larger
# than the physical crystal radius so it only acts as a phi/z mask.
const SEGMENT_SELECTOR_RADIUS = R_OUTER + 1.0

const OUTDIR = @__DIR__

println()
println("============================================================")
println(" Building pure cylindrical rounded-core HPGe detector")
println(" Crystal:              $CRYSTAL_NAME")
println(" Serial:               $SERIAL")
println(" Depletion voltage:    $VD V")
println(" Operating voltage:    $VOP V")
println(" Temperature:          $TEMPERATURE K")
println(" Outer radius:         $R_OUTER mm")
println(" Crystal height:       $CRYSTAL_HEIGHT mm")
println(" Outer contact shell:  $OUTER_CONTACT_THICKNESS mm")
println(" Core cylinder height: $CORE_CYL_HEIGHT mm")
println(" Core cap radius:      $CORE_CAP_RADIUS mm")
println(" Impurity at z=0:      $(IMP_BOTTOM)e10 cm^-3")
println(" Impurity at z=90:     $(IMP_TOP)e10 cm^-3")
println(" Impurity gradient:    $IMP_GRADIENT_CM4 cm^-4")
println(" Taper:                removed")
println(" Octagonal sides:      removed")
println("============================================================")
println()

# ============================================================
# YAML helper functions
# ============================================================

function core_bore_yaml(indent::Int)
    sp = repeat(" ", indent)

    return """
$(sp)union:
$(sp)  - tube:
$(sp)      r: $(CORE_RADIUS)
$(sp)      h: $(CORE_CYL_HEIGHT)
$(sp)      translate: {z: $(CORE_CYL_CENTER_Z)}

$(sp)  - sphere:
$(sp)      r: $(CORE_CAP_RADIUS)
$(sp)      translate: {z: $(CORE_CAP_CENTER_Z)}
"""
end


function core_bore_item_yaml(indent::Int)
    sp = repeat(" ", indent)

    return """
$(sp)- union:
$(sp)    - tube:
$(sp)        r: $(CORE_RADIUS)
$(sp)        h: $(CORE_CYL_HEIGHT)
$(sp)        translate: {z: $(CORE_CYL_CENTER_Z)}

$(sp)    - sphere:
$(sp)        r: $(CORE_CAP_RADIUS)
$(sp)        translate: {z: $(CORE_CAP_CENTER_Z)}
"""
end


# Full cylindrical semiconductor shape with the rounded core bore removed.
function cylindrical_crystal_yaml(
    indent::Int,
    radius::Float64
)
    sp = repeat(" ", indent)

    return """
$(sp)difference:
$(sp)  - tube:
$(sp)      r: $(radius)
$(sp)      h: $(CRYSTAL_HEIGHT)
$(sp)      translate: {z: $(CRYSTAL_CENTER_Z)}
$(core_bore_item_yaml(indent + 2))
"""
end


function cylindrical_crystal_item_yaml(
    indent::Int,
    radius::Float64
)
    sp = repeat(" ", indent)

    return """
$(sp)- difference:
$(sp)    - tube:
$(sp)        r: $(radius)
$(sp)        h: $(CRYSTAL_HEIGHT)
$(sp)        translate: {z: $(CRYSTAL_CENTER_Z)}
$(core_bore_item_yaml(indent + 4))
"""
end


function outer_shape_yaml(indent::Int)
    return cylindrical_crystal_yaml(
        indent,
        R_OUTER
    )
end


function outer_shape_item_yaml(indent::Int)
    return cylindrical_crystal_item_yaml(
        indent,
        R_OUTER
    )
end


function inner_offset_shape_item_yaml(indent::Int)
    return cylindrical_crystal_item_yaml(
        indent,
        R_INNER_CONTACT
    )
end


# Each segment is:
#
#   phi/z selector intersection
#   (outer cylindrical crystal - inner cylindrical crystal)
#
# This produces a constant 0.5 mm radial shell on the outer mantle.
function outer_segment_yaml(
    name,
    id,
    phifrom,
    phito,
    zcenter,
    height
)
    return """
      - name: $name
        id: $id
        potential: 0.0
        geometry:
          intersection:
            # Oversized selector defining only the segment's phi/z range.
            - tube:
                r: {from: 0.0, to: $(SEGMENT_SELECTOR_RADIUS)}
                phi: {from: $phifrom, to: $phito}
                h: $height
                translate: {z: $zcenter}

            # Constant radial-thickness outer contact shell.
            - difference:
$(outer_shape_item_yaml(16))
$(inner_offset_shape_item_yaml(16))
"""
end

# ============================================================
# Contact YAML
# ============================================================

contacts_yaml = ""

# Front/top contacts: z = 70 to 90 mm, IDs 1-4.
contacts_yaml *= outer_segment_yaml(
    "Front_Q1", 1, 90.0, 180.0, 80.0, 20.0
)
contacts_yaml *= outer_segment_yaml(
    "Front_Q2", 2, 180.0, 270.0, 80.0, 20.0
)
contacts_yaml *= outer_segment_yaml(
    "Front_Q3", 3, 270.0, 360.0, 80.0, 20.0
)
contacts_yaml *= outer_segment_yaml(
    "Front_Q4", 4, 0.0, 90.0, 80.0, 20.0
)

# Back/bottom contacts: z = 0 to 70 mm, IDs 5-8.
contacts_yaml *= outer_segment_yaml(
    "Back_Q1", 5, 90.0, 180.0, 35.0, 70.0
)
contacts_yaml *= outer_segment_yaml(
    "Back_Q2", 6, 180.0, 270.0, 35.0, 70.0
)
contacts_yaml *= outer_segment_yaml(
    "Back_Q3", 7, 270.0, 360.0, 35.0, 70.0
)
contacts_yaml *= outer_segment_yaml(
    "Back_Q4", 8, 0.0, 90.0, 35.0, 70.0
)

# Rounded core electrode matching the rounded bore.
contacts_yaml *= """
      - name: Core
        id: 9
        potential: $(VOP)
        geometry:
$(core_bore_yaml(10))
"""

# ============================================================
# Full YAML configuration
# ============================================================

yaml_config = """
name: Gimpy_B_Pure_Cylinder_Rounded_Core

units:
  length: mm
  angle: deg
  potential: V
  temperature: K

grid:
  coordinates: cartesian
  axes:
    x: {from: -35.0, to: 35.0}
    y: {from: -35.0, to: 35.0}
    z: {from: -5.0, to: 95.0}

medium: vacuum

detectors:
  - name: Gimpy_B_Pure_Cylinder_$(SERIAL)

    semiconductor:
      material: HPGe
      temperature: $(TEMPERATURE)

      impurity_density:
        name: linear
        offset: $(IMP_OFFSET_CM3)cm^-3
        gradient:
          z: $(IMP_GRADIENT_CM4)cm^-4

      geometry:
$(outer_shape_yaml(8))

    contacts:
$(contacts_yaml)
"""

yaml_path = joinpath(
    OUTDIR,
    "gimpy_B_pure_cylinder_rounded_core.yaml"
)

open(yaml_path, "w") do io
    write(io, yaml_config)
end

println("Wrote YAML:")
println("  $yaml_path")

# ============================================================
# Optional YAML preview
# ============================================================

println()
println("YAML preview, first 100 lines:")

for (i, line) in enumerate(eachline(yaml_path))
    i > 100 && break
    println(lpad(i, 4), " | ", line)
end

# ============================================================
# Load simulation
# ============================================================

println()
println("Loading Simulation...")
sim = Simulation(yaml_path)
println("Simulation loaded.")

contact_ids = sort([contact.id for contact in sim.detector.contacts])
println("Detector contact IDs: $contact_ids")

expected_contact_ids = collect(1:9)

contact_ids == expected_contact_ids ||
    error(
        "Unexpected detector contacts. " *
        "Expected $expected_contact_ids, found $contact_ids"
    )

# ============================================================
# Electric potential
# ============================================================

if RUN_ELECTRIC_POTENTIAL
    println()
    println("Calculating electric potential...")

    @time calculate_electric_potential!(
        sim,
        refinement_limits = REFINEMENT_LIMITS
    )

    println("Electric potential calculated.")
else
    println("Skipping electric potential.")
end

# ============================================================
# Electric field
# ============================================================

if RUN_ELECTRIC_FIELD
    println()
    println("Calculating electric field...")
    calculate_electric_field!(sim)
    println("Electric field calculated.")
else
    println("Skipping electric field.")
end

# ============================================================
# Weighting potentials
# ============================================================

if RUN_WEIGHTING_POTENTIALS
    println()
    println("Calculating weighting potentials...")

    for contact_id in CONTACTS_FOR_WEIGHTING
        println("  Contact ID = $contact_id")

        @time calculate_weighting_potential!(
            sim,
            contact_id,
            refinement_limits = REFINEMENT_LIMITS
        )
    end

    println("Weighting potentials calculated.")
else
    println("Skipping weighting potentials.")
end

# ============================================================
# Save simulation
# ============================================================

sim_save_path = joinpath(
    OUTDIR,
    "gimpy_B_pure_cylinder_rounded_core_sim.jls"
)

println()
println("Saving serialized simulation...")
serialize(sim_save_path, sim)

println("Saved simulation:")
println("  $sim_save_path")

println()
println("============================================================")
println("Done.")
println("YAML file:")
println("  $yaml_path")
println("Serialized simulation:")
println("  $sim_save_path")
println("============================================================")