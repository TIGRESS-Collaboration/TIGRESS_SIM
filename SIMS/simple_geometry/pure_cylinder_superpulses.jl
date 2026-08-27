# ============================================================
# pure_cylinder_superpulses.jl
#
# Reads energy-deposition hits from the pure-cylinder Co-60 Geant4
# simulation, selects full-energy 1332.5-keV events, simulates the
# complete multi-site event with SolidStateDetectors.jl, applies
# single-segment quality cuts, aligns accepted waveforms, and averages
# one nine-channel superpulse for each collecting outer contact.
#
# Input CSV columns:
#   event_id,track_id,x_mm,y_mm,z_ssd_mm,edep_keV
#
# Required SSD simulation:
#   gimpy_B_pure_cylinder_rounded_core_sim.jls
#
# Outputs:
#   pure_cylinder_co60_ssd_superpulses.jls
#   pure_cylinder_co60_ssd_superpulses_contact1.jls ... contact8.jls
#   contact plots, event summary, amplitude summary, diagnostics
# ============================================================

using SolidStateDetectors
using Serialization
using Unitful
using Statistics
using Random

ENV["GKSwstype"] = "100"
using Plots

gr()
closeall()

default(
    background_color = :white,
    background_color_inside = :white,
    foreground_color = :black,
    dpi = 150
)

# ============================================================
# User settings
# ============================================================

const SCRIPT_DIR = @__DIR__

const SIM_FILE = joinpath(
    SCRIPT_DIR,
    "gimpy_B_pure_cylinder_rounded_core_sim.jls"
)

const GEANT_HITS_FILE =
    "/data1/flerner/hpge_sims/gimpy_B_pure_cylinder_co60_geant4/build/geant_hits.csv"

const OUT_DIR = joinpath(
    SCRIPT_DIR,
    "pure_cylinder_co60_superpulses"
)

const OUT_PREFIX = "pure_cylinder_co60_ssd_superpulses"

const SUMMARY_CSV = joinpath(
    OUT_DIR,
    "$(OUT_PREFIX)_event_summary.csv"
)

const AMP_SUMMARY_CSV = joinpath(
    OUT_DIR,
    "$(OUT_PREFIX)_amplitude_summary.csv"
)

const DIAGNOSTIC_CSV = joinpath(
    OUT_DIR,
    "$(OUT_PREFIX)_contact_diagnostics.csv"
)

# ============================================================
# SSD charge-drift settings
# ============================================================

const PHI110_DEG = 45.0
const TEMPERATURE_K = 103.0

const DRIFT_DT = 2u"ns"
const MAX_NSTEPS = 20_000

# Start with false/false for the pristine detector-response baseline.
# Enable later for a controlled comparison.
const RUN_DIFFUSION = false
const RUN_SELF_REPULSION = false

# ============================================================
# Output waveform grid
# ============================================================

const OUTPUT_DT_NS = 2.0
const OUTPUT_TRACE_LENGTH_NS = 1200.0
const OUTPUT_TIME_NS = collect(
    0.0:OUTPUT_DT_NS:OUTPUT_TRACE_LENGTH_NS
)
const NSAMPLES = length(OUTPUT_TIME_NS)

# ============================================================
# Contacts
# ============================================================

const OUTER_CONTACT_IDS = collect(1:8)
const CORE_CONTACT_ID = 9
const ALL_CONTACT_IDS = collect(1:9)
const TARGET_CONTACTS = collect(1:8)
const NCONTACTS = length(ALL_CONTACT_IDS)

# Desired number of accepted events in each target-contact average.
const N_ACCEPTED_PER_CONTACT = 300

# Optional upper limit on inspected Geant4 events.
const MAX_GEANT_EVENTS_TO_PROCESS = typemax(Int)

# ============================================================
# Geant4 selection
# ============================================================

const PHOTOPEAK_ENERGY_KEV = 1332.5

# Gate on the sum of all retained Geant4 deposits in an event.
const APPLY_GEANT_ENERGY_GATE = true
const GEANT_ENERGY_TOLERANCE_KEV = 20.0

# Geant4 now writes every positive deposit. This threshold removes tiny
# deposits before creating the SSD event. Set to 0.0 to retain all hits.
const MIN_HIT_EDEP_KEV = 0.0

# The new crystal has no taper, so no taper-boundary filter is required.

# ============================================================
# SSD event-quality selection
# ============================================================

# The collecting outer contact must finish near 1332.5 keV.
const APPLY_TARGET_FINAL_GATE = true
const TARGET_FINAL_TOLERANCE_KEV = 30.0

# The core magnitude must finish near 1332.5 keV.
const APPLY_CORE_FINAL_GATE = true
const CORE_FINAL_TOLERANCE_KEV = 50.0

# One outer segment must dominate the final collected charge.
const APPLY_SEGMENT_CONFIDENCE_CUT = true
const MIN_SEGMENT_CONFIDENCE = 0.95

# Require the largest non-target final residual to remain small.
const APPLY_NON_TARGET_FINAL_CUT = false
const MAX_NON_TARGET_FINAL_KEV = 50.0

# ============================================================
# Trigger alignment and baseline processing
# ============================================================

const APPLY_TRIGGER_ALIGNMENT = true
const TRIGGER_CONTACT_ID = CORE_CONTACT_ID
const TRIGGER_FRACTION = 0.10
const TRIGGER_TIME_NS = 300.0

const BASELINE_START_NS = 0.0
const BASELINE_END_NS = 50.0

# Preserve the native SSD core polarity in saved outputs by default.
# Set true only if you want the simulated core plotted positive.
const FLIP_CORE_POLARITY = false

# ============================================================
# Optional electronics and noise
# ============================================================

const APPLY_PREAMP_RESPONSE = false
const APPLY_INTEGRAL_CROSSTALK = false
const APPLY_DIFFERENTIAL_CROSSTALK = false
const ADD_NOISE = false

const CORE_PREAMP_RISE_10_90_NS = 40.0f0
const OUTER_PREAMP_RISE_10_90_NS = 100.0f0
const CORE_PREAMP_DECAY_TAU_NS = 50_000.0f0
const OUTER_PREAMP_DECAY_TAU_NS = 50_000.0f0

const INTEGRAL_CROSSTALK_COEFF = 0.005f0
const DIFFERENTIAL_CROSSTALK_COEFF_NS = 0.10f0

const NOISE_RMS_KEV = 1.0f0
const RANDOM_SEED = 12345

# Save every accepted event waveform for later outlier diagnostics.
const SAVE_ACCEPTED_EVENT_WAVEFORMS = true

# ============================================================
# Main
# ============================================================

function main()
    Random.seed!(RANDOM_SEED)
    mkpath(OUT_DIR)
    validate_settings()

    println()
    println("============================================================")
    println(" Pure-cylinder Co-60 Geant4 hits -> SSD superpulses")
    println("============================================================")
    println("SSD simulation:")
    println("  ", SIM_FILE)
    println("Geant4 hit file:")
    println("  ", GEANT_HITS_FILE)
    println("Output directory:")
    println("  ", OUT_DIR)
    println("Requested accepted events per contact:")
    println("  ", N_ACCEPTED_PER_CONTACT)
    println()

    sim = deserialize(SIM_FILE)
    T = SolidStateDetectors.get_precision_type(sim)

    println("SSD precision type: ", T)
    println("Attaching ADL charge-drift model...")

    charge_drift_model = ADLChargeDriftModel(
        T = T,
        temperature = TEMPERATURE_K,
        phi110 = PHI110_DEG
    )

    sim.detector = SolidStateDetector(
        sim.detector,
        charge_drift_model
    )

    check_contacts(sim)

    println("Reading Geant4 hits...")
    events, parser_stats = read_geant_hits(GEANT_HITS_FILE)
    event_ids = sort(collect(keys(events)))

    println("Events containing retained Geant4 hits: ", length(event_ids))
    println("Retained Geant4 hit rows: ", parser_stats["retained_hits"])
    println("Discarded sub-threshold hit rows: ", parser_stats["discarded_small_hits"])

    accum = Dict(
        cid => zeros(Float32, NCONTACTS, NSAMPLES)
        for cid in TARGET_CONTACTS
    )

    counts = Dict(
        cid => 0
        for cid in TARGET_CONTACTS
    )

    accepted_waveforms = Dict(
        cid => Matrix{Float32}[]
        for cid in TARGET_CONTACTS
    )

    accepted_event_ids = Dict(
        cid => Int[]
        for cid in TARGET_CONTACTS
    )

    selected_by_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    target_gate_fail = Dict(cid => 0 for cid in TARGET_CONTACTS)
    core_gate_fail = Dict(cid => 0 for cid in TARGET_CONTACTS)
    confidence_fail = Dict(cid => 0 for cid in TARGET_CONTACTS)
    non_target_fail = Dict(cid => 0 for cid in TARGET_CONTACTS)
    already_full = Dict(cid => 0 for cid in TARGET_CONTACTS)

    initialize_summary_csv()
    initialize_amplitude_csv()

    n_processed = 0
    n_geant_energy_fail = 0
    n_ssd_fail = 0
    n_missing_waveform = 0
    n_accepted = 0

    for geant_event_id in event_ids
        n_processed += 1

        if n_processed > MAX_GEANT_EVENTS_TO_PROCESS
            break
        end

        hits = events[geant_event_id]
        isempty(hits) && continue

        total_geant_edep = sum(
            hit.edep_keV
            for hit in hits
        )

        if APPLY_GEANT_ENERGY_GATE &&
           abs(total_geant_edep - PHOTOPEAK_ENERGY_KEV) >
               GEANT_ENERGY_TOLERANCE_KEV

            n_geant_energy_fail += 1

            append_event_summary!(
                geant_event_id,
                total_geant_edep,
                length(hits),
                "geant_energy_gate_fail",
                0,
                0,
                NaN,
                NaN,
                NaN,
                NaN,
                NaN,
                NaN
            )

            continue
        end

        try
            event = make_ssd_event_from_hits(T, hits)

            simulate!(
                event,
                sim,
                Δt = DRIFT_DT,
                max_nsteps = MAX_NSTEPS,
                diffusion = RUN_DIFFUSION,
                self_repulsion = RUN_SELF_REPULSION,
                signal_unit = u"keV",
                end_drift_when_no_field = false,
                geometry_check = true,
                verbose = false
            )

            signals = extract_waveforms(event, sim)

            if !all(
                haskey(signals, cid)
                for cid in ALL_CONTACT_IDS
            )
                n_missing_waveform += 1

                append_event_summary!(
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "missing_waveform",
                    0,
                    0,
                    NaN,
                    NaN,
                    NaN,
                    NaN,
                    NaN,
                    NaN
                )

                continue
            end

            raw_finals = final_amplitudes(signals)

            best_cid,
            second_cid,
            best_final,
            second_final,
            confidence = identify_collecting_segment(raw_finals)

            core_final = raw_finals[CORE_CONTACT_ID]
            total_outer_final = sum(
                raw_finals[cid]
                for cid in OUTER_CONTACT_IDS
            )

            largest_non_target_cid,
            largest_non_target_final =
                largest_non_target_final_amplitude(
                    raw_finals,
                    best_cid
                )

            if best_cid in TARGET_CONTACTS
                selected_by_contact[best_cid] += 1
            end

            status = "accepted"

            if !(best_cid in TARGET_CONTACTS)
                status = "best_not_target"

            elseif APPLY_TARGET_FINAL_GATE &&
                   abs(best_final - PHOTOPEAK_ENERGY_KEV) >
                       TARGET_FINAL_TOLERANCE_KEV

                status = "target_final_gate_fail"
                target_gate_fail[best_cid] += 1

            elseif APPLY_CORE_FINAL_GATE &&
                   abs(abs(core_final) - PHOTOPEAK_ENERGY_KEV) >
                       CORE_FINAL_TOLERANCE_KEV

                status = "core_final_gate_fail"
                core_gate_fail[best_cid] += 1

            elseif APPLY_SEGMENT_CONFIDENCE_CUT &&
                   confidence < MIN_SEGMENT_CONFIDENCE

                status = "confidence_fail"
                confidence_fail[best_cid] += 1

            elseif APPLY_NON_TARGET_FINAL_CUT &&
                   abs(largest_non_target_final) >
                       MAX_NON_TARGET_FINAL_KEV

                status = "non_target_final_fail"
                non_target_fail[best_cid] += 1

            elseif counts[best_cid] >= N_ACCEPTED_PER_CONTACT
                status = "target_already_full"
                already_full[best_cid] += 1
            end

            if status != "accepted"
                append_event_summary!(
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    status,
                    best_cid,
                    second_cid,
                    best_final,
                    second_final,
                    confidence,
                    core_final,
                    total_outer_final,
                    largest_non_target_final
                )

                continue
            end

            waveform_matrix = build_fixed_waveform_matrix(signals)

            if APPLY_TRIGGER_ALIGNMENT
                waveform_matrix =
                    align_waveform_matrix_to_fraction_trigger(
                        waveform_matrix,
                        TRIGGER_CONTACT_ID,
                        fraction = TRIGGER_FRACTION,
                        trigger_time_ns = TRIGGER_TIME_NS
                    )
            end

            baseline_subtract_waveform_matrix!(
                waveform_matrix,
                baseline_start_ns = BASELINE_START_NS,
                baseline_end_ns = BASELINE_END_NS
            )

            if FLIP_CORE_POLARITY
                core_index = cid_to_index(CORE_CONTACT_ID)
                waveform_matrix[core_index, :] .*= -1
            end

            if APPLY_PREAMP_RESPONSE
                apply_preamp_response!(waveform_matrix)
            end

            if APPLY_INTEGRAL_CROSSTALK
                apply_integral_crosstalk!(waveform_matrix)
            end

            if APPLY_DIFFERENTIAL_CROSSTALK
                apply_differential_crosstalk!(waveform_matrix)
            end

            if ADD_NOISE
                waveform_matrix .+=
                    Float32(NOISE_RMS_KEV) .* randn(
                        Float32,
                        size(waveform_matrix)
                    )
            end

            accum[best_cid] .+= waveform_matrix
            counts[best_cid] += 1
            n_accepted += 1

            if SAVE_ACCEPTED_EVENT_WAVEFORMS
                push!(
                    accepted_waveforms[best_cid],
                    copy(waveform_matrix)
                )

                push!(
                    accepted_event_ids[best_cid],
                    geant_event_id
                )
            end

            append_event_summary!(
                geant_event_id,
                total_geant_edep,
                length(hits),
                "accepted",
                best_cid,
                second_cid,
                best_final,
                second_final,
                confidence,
                core_final,
                total_outer_final,
                largest_non_target_final
            )

            println(
                "accepted event ",
                geant_event_id,
                " -> contact ",
                best_cid,
                " count ",
                counts[best_cid],
                "/",
                N_ACCEPTED_PER_CONTACT,
                " total Geant energy = ",
                round(total_geant_edep, digits = 3),
                " keV"
            )

            if all(
                counts[cid] >= N_ACCEPTED_PER_CONTACT
                for cid in TARGET_CONTACTS
            )
                println("All target contacts reached requested statistics.")
                break
            end

        catch error_object
            n_ssd_fail += 1

            @warn(
                "SSD simulation failed for Geant4 event $geant_event_id",
                exception = (
                    error_object,
                    catch_backtrace()
                )
            )

            append_event_summary!(
                geant_event_id,
                total_geant_edep,
                length(hits),
                "ssd_simulation_failed",
                0,
                0,
                NaN,
                NaN,
                NaN,
                NaN,
                NaN,
                NaN
            )
        end
    end

    superpulses = build_and_save_superpulses(
        accum,
        counts
    )

    output = Dict(
        "superpulses" => superpulses,
        "counts" => counts,
        "time_ns" => Float32.(OUTPUT_TIME_NS),
        "contact_ids" => Int32.(ALL_CONTACT_IDS),
        "metadata" => Dict(
            "simulation_file" => SIM_FILE,
            "geant_hits_file" => GEANT_HITS_FILE,
            "detector_geometry" => "pure cylinder with rounded blind core",
            "photopeak_energy_keV" => Float32(PHOTOPEAK_ENERGY_KEV),
            "geant_energy_tolerance_keV" => Float32(GEANT_ENERGY_TOLERANCE_KEV),
            "minimum_hit_edep_keV" => Float32(MIN_HIT_EDEP_KEV),
            "target_final_tolerance_keV" => Float32(TARGET_FINAL_TOLERANCE_KEV),
            "core_final_tolerance_keV" => Float32(CORE_FINAL_TOLERANCE_KEV),
            "minimum_segment_confidence" => Float32(MIN_SEGMENT_CONFIDENCE),
            "diffusion" => RUN_DIFFUSION,
            "self_repulsion" => RUN_SELF_REPULSION,
            "trigger_alignment" => APPLY_TRIGGER_ALIGNMENT,
            "trigger_fraction" => Float32(TRIGGER_FRACTION),
            "trigger_time_ns" => Float32(TRIGGER_TIME_NS),
            "output_dt_ns" => Float32(OUTPUT_DT_NS),
            "output_trace_length_ns" => Float32(OUTPUT_TRACE_LENGTH_NS),
            "core_polarity_flipped" => FLIP_CORE_POLARITY,
            "preamp_response" => APPLY_PREAMP_RESPONSE,
            "integral_crosstalk" => APPLY_INTEGRAL_CROSSTALK,
            "differential_crosstalk" => APPLY_DIFFERENTIAL_CROSSTALK,
            "noise_added" => ADD_NOISE
        )
    )

    combined_file = joinpath(
        OUT_DIR,
        "$(OUT_PREFIX).jls"
    )

    serialize(combined_file, output)

    if SAVE_ACCEPTED_EVENT_WAVEFORMS
        accepted_file = joinpath(
            OUT_DIR,
            "$(OUT_PREFIX)_accepted_waveforms.jls"
        )

        serialize(
            accepted_file,
            Dict(
                "accepted_waveforms" => accepted_waveforms,
                "accepted_geant_event_ids" => accepted_event_ids,
                "time_ns" => Float32.(OUTPUT_TIME_NS),
                "contact_ids" => Int32.(ALL_CONTACT_IDS),
                "counts" => counts,
                "metadata" => output["metadata"]
            )
        )

        println("Saved accepted event waveforms:")
        println("  ", accepted_file)
    end

    write_contact_diagnostics!(
        selected_by_contact,
        counts,
        target_gate_fail,
        core_gate_fail,
        confidence_fail,
        non_target_fail,
        already_full
    )

    println()
    println("============================================================")
    println("Done.")
    println("Processed Geant4 events:       ", n_processed)
    println("Accepted SSD events:           ", n_accepted)
    println("Geant energy-gate failures:    ", n_geant_energy_fail)
    println("Missing-waveform failures:     ", n_missing_waveform)
    println("SSD simulation failures:       ", n_ssd_fail)
    println()
    println("Accepted counts by contact:")

    for cid in TARGET_CONTACTS
        println("  contact ", cid, ": ", counts[cid])
    end

    println()
    println("Saved combined superpulse file:")
    println("  ", combined_file)
    println("============================================================")
end

# ============================================================
# Geant4 hit parsing
# ============================================================

function read_geant_hits(path)
    events = Dict{Int, Vector{NamedTuple}}()

    stats = Dict(
        "rows" => 0,
        "retained_hits" => 0,
        "discarded_small_hits" => 0
    )

    open(path, "r") do io
        header = readline(io)
        columns = strip.(split(header, ","))
        column_index = Dict(
            column => index
            for (index, column) in enumerate(columns)
        )

        required_columns = [
            "event_id",
            "track_id",
            "x_mm",
            "y_mm",
            "z_ssd_mm",
            "edep_keV"
        ]

        for column in required_columns
            haskey(column_index, column) ||
                error(
                    "Missing column '$column' in $path. " *
                    "Found columns: $columns"
                )
        end

        for line in eachline(io)
            isempty(strip(line)) && continue
            stats["rows"] += 1

            parts = strip.(split(line, ","))

            event_id = parse(
                Int,
                parts[column_index["event_id"]]
            )

            track_id = parse(
                Int,
                parts[column_index["track_id"]]
            )

            x_mm = parse(
                Float64,
                parts[column_index["x_mm"]]
            )

            y_mm = parse(
                Float64,
                parts[column_index["y_mm"]]
            )

            z_mm = parse(
                Float64,
                parts[column_index["z_ssd_mm"]]
            )

            edep_keV = parse(
                Float64,
                parts[column_index["edep_keV"]]
            )

            if edep_keV < MIN_HIT_EDEP_KEV
                stats["discarded_small_hits"] += 1
                continue
            end

            hit = (
                event_id = event_id,
                track_id = track_id,
                x_mm = x_mm,
                y_mm = y_mm,
                z_mm = z_mm,
                edep_keV = edep_keV
            )

            push!(
                get!(events, event_id, NamedTuple[]),
                hit
            )

            stats["retained_hits"] += 1
        end
    end

    return events, stats
end

# ============================================================
# Geant4 hits -> SSD event
# ============================================================

function make_ssd_event_from_hits(T, hits)
    positions = CartesianPoint{T}[]
    energies = T[]

    for hit in hits
        push!(
            positions,
            CartesianPoint{T}(
                T(hit.x_mm * 1e-3),
                T(hit.y_mm * 1e-3),
                T(hit.z_mm * 1e-3)
            )
        )

        push!(energies, T(hit.edep_keV))
    end

    return Event(
        positions,
        energies * u"keV"
    )
end

# ============================================================
# Waveform extraction
# ============================================================

function extract_waveforms(event, sim)
    signals = Dict{Int, Any}()
    contact_ids = [contact.id for contact in sim.detector.contacts]

    :waveforms in propertynames(event) ||
        error("SSD event has no waveforms property.")

    for index in eachindex(event.waveforms)
        waveform = event.waveforms[index]

        if waveform !== missing
            signals[contact_ids[index]] = waveform
        end
    end

    return signals
end


function waveform_to_xy(waveform)
    raw_time = collect(getfield(waveform, 1))
    raw_amplitude = collect(getfield(waveform, 2))

    time_ns = try
        Float64.(ustrip.(u"ns", raw_time))
    catch
        Float64.(raw_time)
    end

    amplitude_keV = try
        Float64.(ustrip.(u"keV", raw_amplitude))
    catch
        try
            Float64.(ustrip.(raw_amplitude))
        catch
            Float64.(raw_amplitude)
        end
    end

    nvalues = min(
        length(time_ns),
        length(amplitude_keV)
    )

    return (
        time_ns[1:nvalues],
        amplitude_keV[1:nvalues]
    )
end


function final_amplitudes(signals)
    finals = Dict{Int, Float64}()

    for cid in keys(signals)
        _, amplitude = waveform_to_xy(signals[cid])
        finals[cid] = amplitude[end] - amplitude[1]
    end

    return finals
end


function identify_collecting_segment(finals)
    outer_values = [
        (cid, finals[cid])
        for cid in OUTER_CONTACT_IDS
    ]

    sort!(
        outer_values,
        by = item -> item[2],
        rev = true
    )

    best_cid, best_final = outer_values[1]
    second_cid, second_final = outer_values[2]

    confidence =
        (best_final - second_final) /
        max(abs(best_final), 1e-9)

    return (
        best_cid,
        second_cid,
        best_final,
        second_final,
        confidence
    )
end


function largest_non_target_final_amplitude(
    finals,
    best_cid
)
    values = [
        (cid, finals[cid])
        for cid in OUTER_CONTACT_IDS
        if cid != best_cid
    ]

    sort!(
        values,
        by = item -> abs(item[2]),
        rev = true
    )

    return values[1]
end

# ============================================================
# Resampling and processing
# ============================================================

function build_fixed_waveform_matrix(signals)
    waveform_matrix = zeros(
        Float32,
        NCONTACTS,
        NSAMPLES
    )

    for cid in ALL_CONTACT_IDS
        time_ns, amplitude_keV =
            waveform_to_xy(signals[cid])

        interpolated = interp_constant_edges(
            time_ns,
            amplitude_keV,
            OUTPUT_TIME_NS
        )

        waveform_matrix[cid_to_index(cid), :] .=
            Float32.(interpolated)
    end

    return waveform_matrix
end


function cid_to_index(cid)
    index = findfirst(==(cid), ALL_CONTACT_IDS)
    index === nothing && error("Unknown contact ID: $cid")
    return index
end


function interp_constant_edges(x, y, xnew)
    nvalues = length(x)

    nvalues == 0 && return zeros(Float64, length(xnew))
    nvalues == 1 && return fill(Float64(y[1]), length(xnew))

    output = Vector{Float64}(undef, length(xnew))
    j = 1

    for index in eachindex(xnew)
        value = xnew[index]

        if value <= x[1]
            output[index] = y[1]
            continue
        elseif value >= x[end]
            output[index] = y[end]
            continue
        end

        while j < nvalues - 1 && x[j + 1] < value
            j += 1
        end

        fraction =
            (value - x[j]) /
            (x[j + 1] - x[j])

        output[index] =
            y[j] + fraction * (y[j + 1] - y[j])
    end

    return output
end


function align_waveform_matrix_to_fraction_trigger(
    waveform_matrix,
    trigger_cid;
    fraction = TRIGGER_FRACTION,
    trigger_time_ns = TRIGGER_TIME_NS
)
    trigger_index = cid_to_index(trigger_cid)
    trigger_waveform = Float64.(
        waveform_matrix[trigger_index, :]
    )

    initial_value = trigger_waveform[1]
    final_value = trigger_waveform[end]
    delta = final_value - initial_value

    abs(delta) <= 1e-12 && return waveform_matrix

    threshold = initial_value + fraction * delta

    crossing_index = if delta > 0
        findfirst(value -> value >= threshold, trigger_waveform)
    else
        findfirst(value -> value <= threshold, trigger_waveform)
    end

    crossing_index === nothing && return waveform_matrix

    crossing_time_ns = OUTPUT_TIME_NS[crossing_index]

    query_time_ns =
        OUTPUT_TIME_NS .+
        crossing_time_ns .-
        trigger_time_ns

    aligned = zeros(Float32, size(waveform_matrix))

    for cid in ALL_CONTACT_IDS
        contact_index = cid_to_index(cid)

        aligned[contact_index, :] .= Float32.(
            interp_constant_edges(
                OUTPUT_TIME_NS,
                Float64.(waveform_matrix[contact_index, :]),
                query_time_ns
            )
        )
    end

    return aligned
end


function baseline_subtract_waveform_matrix!(
    waveform_matrix;
    baseline_start_ns = BASELINE_START_NS,
    baseline_end_ns = BASELINE_END_NS
)
    baseline_indices = findall(
        time -> baseline_start_ns <= time <= baseline_end_ns,
        OUTPUT_TIME_NS
    )

    isempty(baseline_indices) &&
        error("No samples in the baseline interval.")

    for cid in ALL_CONTACT_IDS
        contact_index = cid_to_index(cid)

        baseline = mean(
            waveform_matrix[
                contact_index,
                baseline_indices
            ]
        )

        waveform_matrix[contact_index, :] .-= baseline
    end

    return waveform_matrix
end

# ============================================================
# Optional electronics
# ============================================================

function apply_preamp_response!(waveform_matrix)
    for cid in ALL_CONTACT_IDS
        contact_index = cid_to_index(cid)

        rise_time = cid == CORE_CONTACT_ID ?
            CORE_PREAMP_RISE_10_90_NS :
            OUTER_PREAMP_RISE_10_90_NS

        decay_time = cid == CORE_CONTACT_ID ?
            CORE_PREAMP_DECAY_TAU_NS :
            OUTER_PREAMP_DECAY_TAU_NS

        waveform_matrix[contact_index, :] .=
            preamp_shape_charge_waveform(
                waveform_matrix[contact_index, :],
                Float32(OUTPUT_DT_NS),
                Float32(rise_time),
                Float32(decay_time)
            )
    end

    return waveform_matrix
end


function preamp_shape_charge_waveform(
    input_waveform,
    dt_ns::Float32,
    rise_10_90_ns::Float32,
    decay_tau_ns::Float32
)
    nsamples = length(input_waveform)
    rise_output = similar(input_waveform)
    final_output = similar(input_waveform)

    rise_tau_ns =
        rise_10_90_ns / Float32(log(9.0))

    if rise_tau_ns <= 0
        rise_output .= input_waveform
    else
        alpha_rise = exp(-dt_ns / rise_tau_ns)
        rise_output[1] = input_waveform[1]

        for index in 2:nsamples
            rise_output[index] =
                alpha_rise * rise_output[index - 1] +
                (1.0f0 - alpha_rise) * input_waveform[index]
        end
    end

    if decay_tau_ns <= 0
        final_output .= rise_output
    else
        alpha_decay = exp(-dt_ns / decay_tau_ns)
        final_output[1] = rise_output[1]

        for index in 2:nsamples
            final_output[index] =
                alpha_decay * final_output[index - 1] +
                rise_output[index] - rise_output[index - 1]
        end
    end

    return final_output
end


function apply_integral_crosstalk!(waveform_matrix)
    mixed = copy(waveform_matrix)

    for receiver in 1:NCONTACTS
        for source in 1:NCONTACTS
            if receiver != source
                mixed[receiver, :] .+=
                    INTEGRAL_CROSSTALK_COEFF .*
                    waveform_matrix[source, :]
            end
        end
    end

    waveform_matrix .= mixed
    return waveform_matrix
end


function apply_differential_crosstalk!(waveform_matrix)
    derivative = zeros(Float32, size(waveform_matrix))

    for channel in 1:NCONTACTS
        for sample in 2:NSAMPLES
            derivative[channel, sample] =
                (
                    waveform_matrix[channel, sample] -
                    waveform_matrix[channel, sample - 1]
                ) / Float32(OUTPUT_DT_NS)
        end
    end

    mixed = copy(waveform_matrix)

    for receiver in 1:NCONTACTS
        for source in 1:NCONTACTS
            if receiver != source
                mixed[receiver, :] .+=
                    DIFFERENTIAL_CROSSTALK_COEFF_NS .*
                    derivative[source, :]
            end
        end
    end

    waveform_matrix .= mixed
    return waveform_matrix
end

# ============================================================
# Superpulse output
# ============================================================

function build_and_save_superpulses(accum, counts)
    superpulses = Dict{Int, Matrix{Float32}}()

    for cid in TARGET_CONTACTS
        if counts[cid] == 0
            println("No accepted events for contact ", cid)
            continue
        end

        superpulse =
            accum[cid] ./ Float32(counts[cid])

        superpulses[cid] = superpulse

        contact_file = joinpath(
            OUT_DIR,
            "$(OUT_PREFIX)_contact$(cid).jls"
        )

        serialize(contact_file, superpulse)
        plot_superpulse(superpulse, cid)
        append_amplitude_summary!(cid, superpulse)

        println(
            "Saved contact ",
            cid,
            " superpulse using ",
            counts[cid],
            " events."
        )
    end

    return superpulses
end


function plot_superpulse(superpulse, target_cid)
    plot_object = plot(
        title =
            "Pure-cylinder SSD superpulse, collecting contact $target_cid",
        xlabel = "time / ns",
        ylabel = "signal / keV",
        size = (1100, 750),
        legend = :outerright,
        xlims = (0.0, 1000.0)
    )

    for cid in ALL_CONTACT_IDS
        contact_index = cid_to_index(cid)

        plot!(
            plot_object,
            OUTPUT_TIME_NS,
            superpulse[contact_index, :],
            label = cid == CORE_CONTACT_ID ?
                "FV/core" :
                "contact $cid",
            linewidth = cid == target_cid ?
                3.0 :
                cid == CORE_CONTACT_ID ?
                2.5 :
                1.3
        )
    end

    output_png = joinpath(
        OUT_DIR,
        "$(OUT_PREFIX)_contact$(target_cid).png"
    )

    savefig(plot_object, output_png)
end


function signed_peak_amplitude(waveform)
    index = argmax(abs.(waveform))
    return Float64(waveform[index])
end


function final_amplitude(waveform)
    return Float64(waveform[end])
end


function initialize_amplitude_csv()
    open(AMP_SUMMARY_CSV, "w") do io
        println(
            io,
            "target_contact,target_peak_keV,target_final_keV," *
            "core_contact,core_peak_keV,core_final_keV," *
            "second_contact,second_peak_keV,second_final_keV," *
            "second_peak_over_target_peak," *
            "second_final_over_target_final"
        )
    end
end


function append_amplitude_summary!(
    target_cid,
    superpulse
)
    target_index = cid_to_index(target_cid)
    core_index = cid_to_index(CORE_CONTACT_ID)

    target_peak = signed_peak_amplitude(
        superpulse[target_index, :]
    )

    target_final = final_amplitude(
        superpulse[target_index, :]
    )

    core_peak = signed_peak_amplitude(
        superpulse[core_index, :]
    )

    core_final = final_amplitude(
        superpulse[core_index, :]
    )

    other_outer = [
        (
            cid,
            signed_peak_amplitude(
                superpulse[cid_to_index(cid), :]
            ),
            final_amplitude(
                superpulse[cid_to_index(cid), :]
            )
        )
        for cid in OUTER_CONTACT_IDS
        if cid != target_cid
    ]

    sort!(
        other_outer,
        by = item -> abs(item[2]),
        rev = true
    )

    second_cid,
    second_peak,
    second_final = other_outer[1]

    peak_ratio =
        second_peak / max(abs(target_peak), 1e-9)

    final_ratio =
        second_final / max(abs(target_final), 1e-9)

    open(AMP_SUMMARY_CSV, "a") do io
        println(
            io,
            join(
                (
                    target_cid,
                    target_peak,
                    target_final,
                    CORE_CONTACT_ID,
                    core_peak,
                    core_final,
                    second_cid,
                    second_peak,
                    second_final,
                    peak_ratio,
                    final_ratio
                ),
                ","
            )
        )
    end
end

# ============================================================
# CSV diagnostics
# ============================================================

function initialize_summary_csv()
    open(SUMMARY_CSV, "w") do io
        println(
            io,
            "geant_event_id,total_geant_edep_keV,n_hits,status," *
            "best_contact,second_contact,best_final_keV," *
            "second_final_keV,confidence,core_final_keV," *
            "total_outer_final_keV,largest_non_target_final_keV"
        )
    end
end


function append_event_summary!(
    geant_event_id,
    total_geant_edep,
    n_hits,
    status,
    best_contact,
    second_contact,
    best_final,
    second_final,
    confidence,
    core_final,
    total_outer_final,
    largest_non_target_final
)
    open(SUMMARY_CSV, "a") do io
        println(
            io,
            join(
                (
                    geant_event_id,
                    total_geant_edep,
                    n_hits,
                    status,
                    best_contact,
                    second_contact,
                    best_final,
                    second_final,
                    confidence,
                    core_final,
                    total_outer_final,
                    largest_non_target_final
                ),
                ","
            )
        )
    end
end


function write_contact_diagnostics!(
    selected,
    accepted,
    target_fail,
    core_fail,
    confidence_fail,
    non_target_fail,
    already_full
)
    open(DIAGNOSTIC_CSV, "w") do io
        println(
            io,
            "contact,selected,accepted,target_final_fail," *
            "core_final_fail,confidence_fail," *
            "non_target_final_fail,target_already_full"
        )

        for cid in TARGET_CONTACTS
            println(
                io,
                join(
                    (
                        cid,
                        selected[cid],
                        accepted[cid],
                        target_fail[cid],
                        core_fail[cid],
                        confidence_fail[cid],
                        non_target_fail[cid],
                        already_full[cid]
                    ),
                    ","
                )
            )
        end
    end
end

# ============================================================
# Validation
# ============================================================

function check_contacts(sim)
    contact_ids = sort([
        contact.id
        for contact in sim.detector.contacts
    ])

    contact_ids == ALL_CONTACT_IDS ||
        error(
            "Expected SSD contacts $ALL_CONTACT_IDS, " *
            "but found $contact_ids"
        )

    println("All expected SSD contacts are present: ", contact_ids)
end


function validate_settings()
    isfile(SIM_FILE) ||
        error("SSD simulation file not found: $SIM_FILE")

    isfile(GEANT_HITS_FILE) ||
        error("Geant4 hit file not found: $GEANT_HITS_FILE")

    N_ACCEPTED_PER_CONTACT > 0 ||
        error("N_ACCEPTED_PER_CONTACT must be positive.")

    GEANT_ENERGY_TOLERANCE_KEV >= 0 ||
        error("GEANT_ENERGY_TOLERANCE_KEV must be nonnegative.")

    TARGET_FINAL_TOLERANCE_KEV >= 0 ||
        error("TARGET_FINAL_TOLERANCE_KEV must be nonnegative.")

    CORE_FINAL_TOLERANCE_KEV >= 0 ||
        error("CORE_FINAL_TOLERANCE_KEV must be nonnegative.")
end

# ============================================================
# Run
# ============================================================

main()