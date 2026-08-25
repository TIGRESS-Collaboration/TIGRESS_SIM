# build_gimpy_B_constant_thickness_rounded_core_offset_bore.jl
# Gimpy B TIGRESS segmented HPGe crystal
#
# Features:
#   - constant-thickness outer contacts
#   - rounded blind-end core bore
#   - offset core bore centered on the tapered/top face
#   - Mikey A impurity gradient
#   - electric potential / electric field calculation
#   - optional weighting potentials
#
# Important geometry update:
#   The core bore is no longer centered at x = 0, y = 0.
#   The bore axis is shifted toward the center of the tapered top face.
#
# Outputs:
#   gimpy_B_constant_thickness_rounded_core_offset_bore.yaml
#   gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls
# ============================================================

using SolidStateDetectors
using Serialization

# ============================================================
# Run switches
# ============================================================

const RUN_ELECTRIC_POTENTIAL = true
const RUN_ELECTRIC_FIELD = true

# Keep false while debugging geometry/field if runtime is too high.
const RUN_WEIGHTING_POTENTIALS = true

# Refinement controls.
# For fast tests: [1.0]
# For better field: [1.0, 0.5, 0.25, 0.1]
const REFINEMENT_LIMITS = [1.0, 0.5, 0.25]

# ============================================================
# Mikey A / Gimpy B detector parameters
# ============================================================

const CRYSTAL_NAME = "Gimpy_B"
const SERIAL = 74037

const VD = 2500.0
const VOP = 3500.0
const TEMPERATURE = 103.0

# Impurity values in units of 1e10 cm^-3.
# z = 0 mm is more pure.
# z = 90 mm is less pure.
const IMP_BOTTOM = 0.95
const IMP_TOP = 2.25

const IMP_OFFSET_CM3 = IMP_BOTTOM * 1e10
const IMP_GRADIENT_CM4 = (IMP_TOP - IMP_BOTTOM) * 1e10 / 9.0

# ============================================================
# Geometry parameters in mm
# ============================================================

const R_OUTER = 30.0
const OCT_APOTHEM = 29.0

const CRYSTAL_HEIGHT = 90.0

# Rounded core bore.
const CORE_RADIUS = 5.0
const CORE_HEIGHT = 75.0

# Tube from z = 0 to 70 mm, sphere centered at z = 70 mm,
# sphere reaches z = 75 mm.
const CORE_CAP_RADIUS = CORE_RADIUS
const CORE_CYL_HEIGHT = CORE_HEIGHT - CORE_CAP_RADIUS
const CORE_CYL_CENTER_Z = CORE_CYL_HEIGHT / 2
const CORE_CAP_CENTER_Z = CORE_CYL_HEIGHT

# Taper geometry.
const TAPER_START_Z = 60.0
const TAPER_ANGLE = 22.5
const CUT_BOX_WIDTH = 200.0

# ============================================================
# Core bore offset
# ============================================================
const CORE_DISTANCE_TO_CONTACT34_FLAT_SIDE = 27.2
const CORE_OFFSET_X = OCT_APOTHEM - CORE_DISTANCE_TO_CONTACT34_FLAT_SIDE
const CORE_OFFSET_Y = 0.0

# Constant outer-contact thickness in mm.
const OUTER_CONTACT_THICKNESS = 0.5

const R_INNER_CONTACT = R_OUTER - OUTER_CONTACT_THICKNESS
const OCT_APOTHEM_INNER = OCT_APOTHEM - OUTER_CONTACT_THICKNESS

# Approximate normal offset for the taper plane.
const TAPER_NORMAL_SHIFT = OUTER_CONTACT_THICKNESS / cosd(TAPER_ANGLE)

const CUT_BOX_CENTER_OUTER =
    OCT_APOTHEM + (CUT_BOX_WIDTH / 2) / cosd(TAPER_ANGLE)

const CUT_BOX_CENTER_INNER =
    (OCT_APOTHEM - TAPER_NORMAL_SHIFT) + (CUT_BOX_WIDTH / 2) / cosd(TAPER_ANGLE)

const OUTDIR = @__DIR__

println()
println("============================================================")
println(" Building rounded-core constant-thickness config")
println(" Crystal:      $CRYSTAL_NAME")
println(" Serial:       $SERIAL")
println(" Vd:           $VD V")
println(" Vop:          $VOP V")
println(" Temperature:  $TEMPERATURE K")
println(" Impurity z=0:  $(IMP_BOTTOM)e10 cm^-3")
println(" Impurity z=90: $(IMP_TOP)e10 cm^-3")
println(" Gradient:      $(IMP_GRADIENT_CM4) cm^-4")
println(" Contact thickness: $(OUTER_CONTACT_THICKNESS) mm")
println(" Core bore: tube h=$(CORE_CYL_HEIGHT) mm + sphere cap r=$(CORE_CAP_RADIUS) mm")
println(" Core bore offset: x=$(CORE_OFFSET_X) mm, y=$(CORE_OFFSET_Y) mm")
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
$(sp)      translate: {x: $(CORE_OFFSET_X), y: $(CORE_OFFSET_Y), z: $(CORE_CYL_CENTER_Z)}

$(sp)  - sphere:
$(sp)      r: $(CORE_CAP_RADIUS)
$(sp)      translate: {x: $(CORE_OFFSET_X), y: $(CORE_OFFSET_Y), z: $(CORE_CAP_CENTER_Z)}
"""
end

function core_bore_item_yaml(indent::Int)
    sp = repeat(" ", indent)

    return """
$(sp)- union:
$(sp)    - tube:
$(sp)        r: $(CORE_RADIUS)
$(sp)        h: $(CORE_CYL_HEIGHT)
$(sp)        translate: {x: $(CORE_OFFSET_X), y: $(CORE_OFFSET_Y), z: $(CORE_CYL_CENTER_Z)}

$(sp)    - sphere:
$(sp)        r: $(CORE_CAP_RADIUS)
$(sp)        translate: {x: $(CORE_OFFSET_X), y: $(CORE_OFFSET_Y), z: $(CORE_CAP_CENTER_Z)}
"""
end

function main_mask_yaml(indent::Int, radius::Float64, apothem::Float64)
    sp = repeat(" ", indent)
    width = 2 * apothem

    return """
$(sp)intersection:
$(sp)  - tube:
$(sp)      r: $(radius)
$(sp)      h: $(CRYSTAL_HEIGHT)
$(sp)      translate: {z: $(CRYSTAL_HEIGHT / 2)}

$(sp)  - intersection:
$(sp)      - box:
$(sp)          widths: [$(width), $(width), $(CRYSTAL_HEIGHT)]
$(sp)          origin: [0.0, 0.0, $(CRYSTAL_HEIGHT / 2)]

$(sp)      - box:
$(sp)          widths: [$(width), $(width), $(CRYSTAL_HEIGHT)]
$(sp)          origin: [0.0, 0.0, $(CRYSTAL_HEIGHT / 2)]
$(sp)          rotate:
$(sp)            Z: 45.0
"""
end

function main_mask_item_yaml(indent::Int, radius::Float64, apothem::Float64)
    sp = repeat(" ", indent)
    width = 2 * apothem

    return """
$(sp)- intersection:
$(sp)    - tube:
$(sp)        r: $(radius)
$(sp)        h: $(CRYSTAL_HEIGHT)
$(sp)        translate: {z: $(CRYSTAL_HEIGHT / 2)}

$(sp)    - intersection:
$(sp)        - box:
$(sp)            widths: [$(width), $(width), $(CRYSTAL_HEIGHT)]
$(sp)            origin: [0.0, 0.0, $(CRYSTAL_HEIGHT / 2)]

$(sp)        - box:
$(sp)            widths: [$(width), $(width), $(CRYSTAL_HEIGHT)]
$(sp)            origin: [0.0, 0.0, $(CRYSTAL_HEIGHT / 2)]
$(sp)            rotate:
$(sp)              Z: 45.0
"""
end

function crystal_shape_yaml(
    indent::Int,
    radius::Float64,
    apothem::Float64,
    cut_box_center::Float64
)
    sp = repeat(" ", indent)

    return """
$(sp)difference:
$(main_mask_item_yaml(indent + 2, radius, apothem))
$(core_bore_item_yaml(indent + 2))
$(sp)  - box:
$(sp)      widths: [$(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH)]
$(sp)      origin: [-$(round(cut_box_center, digits = 4)), 0.0, $(TAPER_START_Z)]
$(sp)      rotate:
$(sp)        Y: $(TAPER_ANGLE)

$(sp)  - box:
$(sp)      widths: [$(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH)]
$(sp)      origin: [0.0, $(round(cut_box_center, digits = 4)), $(TAPER_START_Z)]
$(sp)      rotate:
$(sp)        X: $(TAPER_ANGLE)
"""
end

function crystal_shape_item_yaml(
    indent::Int,
    radius::Float64,
    apothem::Float64,
    cut_box_center::Float64
)
    sp = repeat(" ", indent)

    return """
$(sp)- difference:
$(main_mask_item_yaml(indent + 4, radius, apothem))
$(core_bore_item_yaml(indent + 4))
$(sp)    - box:
$(sp)        widths: [$(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH)]
$(sp)        origin: [-$(round(cut_box_center, digits = 4)), 0.0, $(TAPER_START_Z)]
$(sp)        rotate:
$(sp)          Y: $(TAPER_ANGLE)

$(sp)    - box:
$(sp)        widths: [$(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH), $(CUT_BOX_WIDTH)]
$(sp)        origin: [0.0, $(round(cut_box_center, digits = 4)), $(TAPER_START_Z)]
$(sp)        rotate:
$(sp)          X: $(TAPER_ANGLE)
"""
end

function outer_shape_yaml(indent::Int)
    return crystal_shape_yaml(
        indent,
        R_OUTER,
        OCT_APOTHEM,
        CUT_BOX_CENTER_OUTER
    )
end

function outer_shape_item_yaml(indent::Int)
    return crystal_shape_item_yaml(
        indent,
        R_OUTER,
        OCT_APOTHEM,
        CUT_BOX_CENTER_OUTER
    )
end

function inner_offset_shape_item_yaml(indent::Int)
    return crystal_shape_item_yaml(
        indent,
        R_INNER_CONTACT,
        OCT_APOTHEM_INNER,
        CUT_BOX_CENTER_INNER
    )
end

function outer_segment_yaml(name, id, phifrom, phito, zcenter, height)
    return """
      - name: $name
        id: $id
        potential: 0.0
        geometry:
          intersection:
            # Segment selector: quadrant and front/back z range.
            # r = 31 mm is just an oversized selector, not the crystal radius.
            - tube:
                r: {from: 0.0, to: 31.0}
                phi: {from: $phifrom, to: $phito}
                h: $height
                translate: {z: $zcenter}

            # Constant-thickness outer contact:
            # segment selector ∩ (outer shape - inner offset shape)
            - difference:
$(outer_shape_item_yaml(16))
$(inner_offset_shape_item_yaml(16))
"""
end

# ============================================================
# Contact YAML
# ============================================================

contacts_yaml = ""

# Front/top segments: z = 70 to 90 mm, IDs 1-4.
contacts_yaml *= outer_segment_yaml("Front_Q1", 1, 90.0,  180.0, 80.0, 20.0)
contacts_yaml *= outer_segment_yaml("Front_Q2", 2, 180.0, 270.0, 80.0, 20.0)
contacts_yaml *= outer_segment_yaml("Front_Q3", 3, 270.0, 360.0, 80.0, 20.0)
contacts_yaml *= outer_segment_yaml("Front_Q4", 4, 0.0,   90.0,  80.0, 20.0)

# Back/bottom segments: z = 0 to 70 mm, IDs 5-8.
contacts_yaml *= outer_segment_yaml("Back_Q1", 5, 90.0,   180.0, 35.0, 70.0)
contacts_yaml *= outer_segment_yaml("Back_Q2", 6, 180.0, 270.0, 35.0, 70.0)
contacts_yaml *= outer_segment_yaml("Back_Q3", 7, 270.0, 360.0, 35.0, 70.0)
contacts_yaml *= outer_segment_yaml("Back_Q4", 8, 0.0,   90.0,  35.0, 70.0)

# Offset rounded core contact, matching the offset rounded bore.
contacts_yaml *= """
      - name: Core
        id: 9
        potential: $(VOP)
        geometry:
$(core_bore_yaml(10))
"""

# ============================================================
# Full YAML config
# ============================================================

yaml_config = """
name: Gimpy_B_Segmented_Clover_HPGe_Constant_Thickness_Rounded_Core_Offset_Bore

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
  - name: Gimpy_B_Crystal_$(SERIAL)

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

yaml_path = joinpath(OUTDIR, "gimpy_B_constant_thickness_rounded_core_offset_bore.yaml")

open(yaml_path, "w") do io
    write(io, yaml_config)
end

println(" Wrote YAML:")
println("   $yaml_path")

# ============================================================
# Optional YAML preview
# ============================================================

println()
println("YAML preview, first 120 lines:")
for (i, line) in enumerate(eachline(yaml_path))
    i > 120 && break
    println(lpad(i, 4), " | ", line)
end

# ============================================================
# Load simulation
# ============================================================

println()
println(" Loading Simulation...")
sim = Simulation(yaml_path)
println(" Simulation loaded.")

# ============================================================
# Electric potential calculation
# ============================================================

if RUN_ELECTRIC_POTENTIAL
    println()
    println(" Calculating electric potential...")

    @time calculate_electric_potential!(
        sim,
        refinement_limits = REFINEMENT_LIMITS
    )

    println(" Electric potential calculated.")
else
    println("  Skipping electric potential.")
end

# ============================================================
# Electric field calculation
# ============================================================

if RUN_ELECTRIC_FIELD
    println()
    println(" Calculating electric field...")
    calculate_electric_field!(sim)
    println(" Electric field calculated.")
else
    println("  Skipping electric field.")
end

# ============================================================
# Weighting potentials
# ============================================================

const CONTACTS_FOR_WEIGHTING = 1:9

if RUN_WEIGHTING_POTENTIALS
    println()
    println(" Calculating weighting potentials...")

    for contact_id in CONTACTS_FOR_WEIGHTING
        println("   Contact id = $contact_id")

        @time calculate_weighting_potential!(
            sim,
            contact_id,
            refinement_limits = REFINEMENT_LIMITS
        )
    end

    println(" Weighting potentials calculated.")
end

# ============================================================
# Save simulation
# ============================================================

sim_save_path = joinpath(OUTDIR, "gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls")

println()
println(" Saving serialized simulation...")
serialize(sim_save_path, sim)

println(" Saved simulation:")
println("   $sim_save_path")

println()
println("============================================================")
println(" Done.")
println(" YAML file:")
println("   $yaml_path")
println(" Serialized simulation:")
println("   $sim_save_path")
println(" Core bore offset used:")
println("   x = $(CORE_OFFSET_X) mm")
println("   y = $(CORE_OFFSET_Y) mm")
println("============================================================")