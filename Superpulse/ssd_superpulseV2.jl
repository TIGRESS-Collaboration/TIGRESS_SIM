#
# Reads Geant4 energy-deposition points, simulates SSD/ADL
# waveforms, selects clean 1332.5-keV single-segment-like
# events, and averages them into simulated superpulses.
#
# Updated classification improvements:
#   - classify Geant4 events by summed deposited energy per contact
#   - require Geant4 contact and SSD collecting contact to agree
#   - reject multi-contact Geant4 events before SSD waveform averaging
#   - use absolute final amplitudes for SSD collecting-contact logic
#   - optional segment-boundary and taper-boundary filters
#   - accepted-event position diagnostics
# ============================================================

using SolidStateDetectors
using Serialization
using Unitful
using Statistics
using Random
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

const SIM_FILE = "/data1/flerner/hpge_sims/CNN/gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls"
const GEANT_HITS_FILE = "/data1/flerner/hpge_sims/geant/build/geant_hits.csv"

const OUT_PREFIX = "repulsion_gimpy_geant_ssd_superpulses"
const OUT_DIR = "/data1/flerner/hpge_sims/CNN/gimpytests"

const OUT_AMP_SUMMARY_CSV = joinpath(OUT_DIR, "$(OUT_PREFIX)_amplitude_summary.csv")
const OUT_EVENT_SUMMARY_CSV = joinpath(OUT_DIR, "$(OUT_PREFIX)_event_energy_tracks.csv")
const OUT_EVENT_BAR_PLOT = joinpath(OUT_DIR, "$(OUT_PREFIX)_event_energy_tracks.png")
const OUT_ACCEPTED_POSITIONS_CSV = joinpath(OUT_DIR, "$(OUT_PREFIX)_accepted_positions.csv")

const MAX_EVENTS_IN_EVENT_BAR_PLOT = 100

# ADL drift settings.
const PHI110_DEG = 45.0
const TEMPERATURE_K = 103.0

# SSD drift integration settings.
const DRIFT_DT = 2u"ns"
const MAX_NSTEPS = 20_000

const RUN_DIFFUSION = true
const RUN_SELF_REPULSION = true

# Fixed output waveform grid.
const ML_DT_NS = 2.0
const ML_TRACE_LENGTH_NS = 1200.0
const ML_TIME_NS = collect(0.0:ML_DT_NS:ML_TRACE_LENGTH_NS)
const NSAMPLES = length(ML_TIME_NS)

# Contacts 1-8 are outer segments; contact 9 is the core.
const OUTER_CONTACT_IDS = collect(1:8)
const CORE_CONTACT_ID = 9
const ALL_CONTACT_IDS = collect(1:9)
const NCONTACTS = length(ALL_CONTACT_IDS)

# Build one superpulse for each outer collecting contact.
const TARGET_CONTACTS = collect(1:8)

# Number of accepted events averaged into each contact superpulse.
const N_ACCEPTED_PER_CONTACT = 500

# Stop after this many Geant4 events if not enough pass cuts.
const MAX_GEANT_EVENTS_TO_PROCESS = typemax(Int)

# Geant4 total-energy gate.
const APPLY_GEANT_ENERGY_GATE = true
const PHOTOPEAK_ENERGY_KEV = 1332.5
const PHOTOPEAK_TOLERANCE_KEV = 10.0

# Drop tiny Geant4 deposits that often occur near boundaries.
const MIN_HIT_EDEP_KEV = 0.0

# New Geant4 single-contact topology gate.
# For debugging mirror pulses, keep this strict.
const APPLY_GEANT_SINGLE_CONTACT_GATE = true
const MIN_GEANT_CONTACT_FRACTION = 0.90
const MAX_GEANT_SECOND_CONTACT_ENERGY_KEV = 60.0

# Require the Geant4 expected contact and SSD waveform collecting contact to agree.
const REQUIRE_GEANT_SSD_CONTACT_AGREEMENT = true

# Optional segment-boundary filter.
# Useful while debugging contact maps and mirror-pulse signs.
const APPLY_SEGMENT_BOUNDARY_FILTER = false
const PHI_BOUNDARY_MARGIN_DEG = 3.0
const Z_BOUNDARY_MARGIN_MM = 2.0

# Optional taper-boundary filter.
const APPLY_TAPER_BOUNDARY_FILTER = true
const TAPER_BOUNDARY_MARGIN_MM = 2

# Require one outer contact to dominate before averaging.
const MIN_SEGMENT_CONFIDENCE = 0.90

# Keep false when producing pristine SSD superpulses.
const APPLY_PREAMP_RESPONSE = false
const APPLY_INTEGRAL_CROSSTALK = false
const APPLY_DIFFERENTIAL_CROSSTALK = false

# Placeholder electronics values used only if switches above are enabled.
const CORE_PREAMP_RISE_10_90_NS = 40.0f0
const OUTER_PREAMP_RISE_10_90_NS = 100.0f0
const CORE_PREAMP_DECAY_TAU_NS = 50_000.0f0
const OUTER_PREAMP_DECAY_TAU_NS = 50_000.0f0
const IXT_COEFF = 0.005f0
const DXT_COEFF_NS = 0.10f0

# Optional Gaussian noise.
const ADD_NOISE = false
const NOISE_RMS_KEV = 1.0f0
const RANDOM_SEED = 12345

# ============================================================
# Trigger alignment settings
# ============================================================

const APPLY_TRIGGER_ALIGNMENT = true
const TRIGGER_CONTACT_ID = CORE_CONTACT_ID
const TRIGGER_FRACTION = 0.10
const TRIGGER_TIME_NS = 300.0

# SSD reconstructed-amplitude gates.
const APPLY_SSD_TARGET_FINAL_GATE = true
const TARGET_FINAL_TOLERANCE_KEV = 30.0

const APPLY_SSD_CORE_FINAL_GATE = true
const CORE_FINAL_TOLERANCE_KEV = 50.0

# ============================================================
# Main
# ============================================================

function main()
    mkpath(OUT_DIR)
    Random.seed!(RANDOM_SEED)

    println()
    println("============================================================")
    println(" Geant4 hits -> SSD waveforms -> simulated superpulses")
    println(" Updated classification version")
    println("============================================================")
    println("SSD sim file:")
    println("  ", SIM_FILE)
    println("Geant4 hits file:")
    println("  ", GEANT_HITS_FILE)

    sim = deserialize(SIM_FILE)

    T = SolidStateDetectors.get_precision_type(sim)
    println("SSD precision type: ", T)

    println()
    println("Attaching ADLChargeDriftModel")
    println("  temperature = ", TEMPERATURE_K, " K")
    println("  phi110      = ", PHI110_DEG, " deg")

    charge_drift_model = ADLChargeDriftModel(
        T = T,
        temperature = TEMPERATURE_K,
        phi110 = PHI110_DEG
    )

    sim.detector = SolidStateDetector(sim.detector, charge_drift_model)

    println("ADL drift model attached.")

    check_contacts(sim)

    println()
    println("Reading Geant4 hits...")

    events = read_geant_hits(GEANT_HITS_FILE)
    event_ids = sort(collect(keys(events)))

    println("Number of Geant4 events with at least one hit: ", length(event_ids))

    write_event_energy_track_summary!(events, OUT_EVENT_SUMMARY_CSV)

    plot_event_energy_track_bars(
        events,
        OUT_EVENT_BAR_PLOT,
        max_events = MAX_EVENTS_IN_EVENT_BAR_PLOT
    )

    # Running sums for superpulse averaging.
    accum = Dict(cid => zeros(Float32, NCONTACTS, NSAMPLES) for cid in TARGET_CONTACTS)
    counts = Dict(cid => 0 for cid in TARGET_CONTACTS)

    summary_csv = joinpath(OUT_DIR, "$(OUT_PREFIX)_summary.csv")

    open(summary_csv, "w") do io
        println(
            io,
            "geant_event_id,total_geant_edep_keV,n_hits,status,best_contact,second_contact,best_final_keV,second_final_keV,confidence,core_final_keV,total_outer_final_keV,expected_contact"
        )
    end

    open(OUT_AMP_SUMMARY_CSV, "w") do io
        println(
            io,
            "target_contact,target_peak_keV,target_final_keV,core_contact,core_peak_keV,core_final_keV,second_contact,second_peak_keV,second_final_keV,second_peak_over_target_peak_abs,second_final_over_target_final_abs"
        )
    end

    open(OUT_ACCEPTED_POSITIONS_CSV, "w") do io
        println(
            io,
            "geant_event_id,best_contact,total_edep_keV,x_energy_centroid_mm,y_energy_centroid_mm,z_energy_centroid_mm,r_mm,phi_deg,n_hits,geant_contact_fraction,geant_second_contact,geant_second_contact_energy_keV"
        )
    end

    # Global counters.
    n_processed = 0
    n_energy_gate_fail = 0
    n_geant_single_contact_fail = 0
    n_segment_boundary_fail = 0
    n_taper_boundary_fail = 0
    n_ssd_fail = 0
    n_missing_waveforms = 0
    n_confidence_fail = 0
    n_contact_mismatch_fail = 0
    n_accepted_total = 0
    n_outside_semiconductor_fail = 0
    n_other_ssd_fail = 0

    # Per-contact diagnostics.
    best_contact_after_energy_gate = Dict(cid => 0 for cid in TARGET_CONTACTS)
    confidence_fail_by_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    target_already_full_by_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    accepted_by_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    target_final_fail_by_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    core_final_fail_by_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    contact_mismatch_fail_by_expected_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    geant_single_contact_fail_by_expected_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)

    # SSD failure diagnostics by expected contact from Geant4 xyz.
    ssd_fail_by_expected_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)
    outside_semiconductor_fail_by_expected_contact = Dict(cid => 0 for cid in TARGET_CONTACTS)

    for geant_event_id in event_ids
        n_processed += 1

        if n_processed > MAX_GEANT_EVENTS_TO_PROCESS
            break
        end

        hits = events[geant_event_id]

        if isempty(hits)
            continue
        end

        total_geant_edep = sum(h.edep_keV for h in hits)

        expected_contact,
        geant_second_contact,
        geant_best_energy,
        geant_second_energy,
        geant_contact_fraction,
        geant_e_by_contact = geant_contact_energy_summary(hits)

        # Reject events outside selected photopeak window.
        if APPLY_GEANT_ENERGY_GATE
            if abs(total_geant_edep - PHOTOPEAK_ENERGY_KEV) > PHOTOPEAK_TOLERANCE_KEV
                n_energy_gate_fail += 1

                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "geant_energy_gate_fail",
                    expected_contact,
                    geant_second_contact,
                    geant_best_energy,
                    geant_second_energy,
                    geant_contact_fraction,
                    NaN,
                    NaN,
                    expected_contact
                )

                continue
            end
        end

        if expected_contact in TARGET_CONTACTS
            best_contact_after_energy_gate[expected_contact] += 1
        end

        # Reject Geant4 events whose deposited energy is not dominated by one contact.
        if APPLY_GEANT_SINGLE_CONTACT_GATE
            if geant_contact_fraction < MIN_GEANT_CONTACT_FRACTION ||
               geant_second_energy > MAX_GEANT_SECOND_CONTACT_ENERGY_KEV

                n_geant_single_contact_fail += 1

                if expected_contact in TARGET_CONTACTS
                    geant_single_contact_fail_by_expected_contact[expected_contact] += 1
                end

                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "geant_single_contact_gate_fail",
                    expected_contact,
                    geant_second_contact,
                    geant_best_energy,
                    geant_second_energy,
                    geant_contact_fraction,
                    NaN,
                    NaN,
                    expected_contact
                )

                continue
            end
        end

        # Optional phi/z segment-boundary filter.
        if APPLY_SEGMENT_BOUNDARY_FILTER && event_too_close_to_segment_boundary(hits)
            n_segment_boundary_fail += 1

            append_summary!(
                summary_csv,
                geant_event_id,
                total_geant_edep,
                length(hits),
                "segment_boundary_fail",
                expected_contact,
                geant_second_contact,
                geant_best_energy,
                geant_second_energy,
                geant_contact_fraction,
                NaN,
                NaN,
                expected_contact
            )

            continue
        end

        # Optional taper-boundary filter.
        if APPLY_TAPER_BOUNDARY_FILTER &&
           any(h -> h.z_mm >= 60.0 && is_too_close_to_taper_boundary(h, margin_mm = TAPER_BOUNDARY_MARGIN_MM), hits)

            n_taper_boundary_fail += 1

            append_summary!(
                summary_csv,
                geant_event_id,
                total_geant_edep,
                length(hits),
                "taper_boundary_fail",
                expected_contact,
                geant_second_contact,
                geant_best_energy,
                geant_second_energy,
                geant_contact_fraction,
                NaN,
                NaN,
                expected_contact
            )

            continue
        end

        try
            # Convert Geant4 hit positions and energies into one multi-site SSD event.
            evt = make_ssd_event_from_hits(T, hits)

            simulate!(
                evt,
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

            signals = extract_waveforms(evt, sim)

            if !all(haskey(signals, cid) for cid in ALL_CONTACT_IDS)
                n_missing_waveforms += 1

                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "missing_waveform",
                    expected_contact,
                    geant_second_contact,
                    NaN,
                    NaN,
                    NaN,
                    NaN,
                    NaN,
                    expected_contact
                )

                continue
            end

            raw_finals = final_amplitudes(signals)

            best_cid, second_cid, best_val, second_val, conf =
                identify_collecting_segment(raw_finals)

            core_final = raw_finals[CORE_CONTACT_ID]
            total_outer_final = sum(raw_finals[cid] for cid in OUTER_CONTACT_IDS)

            # Require SSD collecting contact to agree with Geant4 contact classification.
            if REQUIRE_GEANT_SSD_CONTACT_AGREEMENT && best_cid != expected_contact
                n_contact_mismatch_fail += 1

                if expected_contact in TARGET_CONTACTS
                    contact_mismatch_fail_by_expected_contact[expected_contact] += 1
                end

                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "geant_ssd_contact_mismatch",
                    best_cid,
                    second_cid,
                    best_val,
                    second_val,
                    conf,
                    core_final,
                    total_outer_final,
                    expected_contact
                )

                continue
            end

            # Require dominant segment to carry selected energy.
            # Use abs(best_val) to avoid polarity convention problems.
            if APPLY_SSD_TARGET_FINAL_GATE
                if abs(abs(best_val) - PHOTOPEAK_ENERGY_KEV) > TARGET_FINAL_TOLERANCE_KEV

                    if best_cid in TARGET_CONTACTS
                        target_final_fail_by_contact[best_cid] += 1
                    end

                    append_summary!(
                        summary_csv,
                        geant_event_id,
                        total_geant_edep,
                        length(hits),
                        "target_final_gate_fail",
                        best_cid,
                        second_cid,
                        best_val,
                        second_val,
                        conf,
                        core_final,
                        total_outer_final,
                        expected_contact
                    )

                    continue
                end
            end

            # Require core magnitude near selected energy.
            if APPLY_SSD_CORE_FINAL_GATE
                if abs(abs(core_final) - PHOTOPEAK_ENERGY_KEV) > CORE_FINAL_TOLERANCE_KEV

                    if best_cid in TARGET_CONTACTS
                        core_final_fail_by_contact[best_cid] += 1
                    end

                    append_summary!(
                        summary_csv,
                        geant_event_id,
                        total_geant_edep,
                        length(hits),
                        "core_final_gate_fail",
                        best_cid,
                        second_cid,
                        best_val,
                        second_val,
                        conf,
                        core_final,
                        total_outer_final,
                        expected_contact
                    )

                    continue
                end
            end

            if !(best_cid in TARGET_CONTACTS)
                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "best_not_target",
                    best_cid,
                    second_cid,
                    best_val,
                    second_val,
                    conf,
                    core_final,
                    total_outer_final,
                    expected_contact
                )

                continue
            end

            if conf < MIN_SEGMENT_CONFIDENCE
                n_confidence_fail += 1

                if best_cid in TARGET_CONTACTS
                    confidence_fail_by_contact[best_cid] += 1
                end

                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "confidence_fail",
                    best_cid,
                    second_cid,
                    best_val,
                    second_val,
                    conf,
                    core_final,
                    total_outer_final,
                    expected_contact
                )

                continue
            end

            if counts[best_cid] >= N_ACCEPTED_PER_CONTACT

                if best_cid in TARGET_CONTACTS
                    target_already_full_by_contact[best_cid] += 1
                end

                append_summary!(
                    summary_csv,
                    geant_event_id,
                    total_geant_edep,
                    length(hits),
                    "target_already_full",
                    best_cid,
                    second_cid,
                    best_val,
                    second_val,
                    conf,
                    core_final,
                    total_outer_final,
                    expected_contact
                )

                continue
            end

            # Resample event waveforms onto common time grid.
            wf_matrix = build_fixed_waveform_matrix(signals)

            if APPLY_TRIGGER_ALIGNMENT
                wf_matrix = align_waveform_matrix_to_fraction_trigger(
                    wf_matrix,
                    TRIGGER_CONTACT_ID,
                    fraction = TRIGGER_FRACTION,
                    trigger_time_ns = TRIGGER_TIME_NS
                )
            end

            baseline_subtract_waveform_matrix!(
                wf_matrix,
                baseline_start_ns = 0.0,
                baseline_end_ns = 50.0
            )

            if APPLY_PREAMP_RESPONSE
                apply_preamp_response!(wf_matrix)
            end

            if APPLY_INTEGRAL_CROSSTALK
                apply_integral_crosstalk!(wf_matrix)
            end

            if APPLY_DIFFERENTIAL_CROSSTALK
                apply_differential_crosstalk!(wf_matrix)
            end

            if ADD_NOISE
                wf_matrix .+= Float32(NOISE_RMS_KEV) .* randn(Float32, size(wf_matrix))
            end

            # Average this accepted event into its collecting-contact superpulse.
            accum[best_cid] .+= wf_matrix
            counts[best_cid] += 1
            n_accepted_total += 1
            accepted_by_contact[best_cid] += 1

            append_summary!(
                summary_csv,
                geant_event_id,
                total_geant_edep,
                length(hits),
                "accepted",
                best_cid,
                second_cid,
                best_val,
                second_val,
                conf,
                core_final,
                total_outer_final,
                expected_contact
            )

            append_accepted_position!(
                OUT_ACCEPTED_POSITIONS_CSV,
                geant_event_id,
                best_cid,
                total_geant_edep,
                hits,
                geant_contact_fraction,
                geant_second_contact,
                geant_second_energy
            )
            if counts[best_cid] % 100 == 0
                println(
                    "accepted event ",
                    geant_event_id,
                    " -> contact ",
                    best_cid,
                    " count ",
                    counts[best_cid],
                    "/",
                    N_ACCEPTED_PER_CONTACT,
                    " total edep = ",
                    round(total_geant_edep, digits = 2),
                    " keV, geant fraction = ",
                    round(geant_contact_fraction, digits = 4)
                )
            end
            
            if all(counts[cid] >= N_ACCEPTED_PER_CONTACT for cid in TARGET_CONTACTS)
                println("All target contacts reached desired statistics.")
                break
            end

        catch err
            n_ssd_fail += 1

            expected_fail_cid = expected_contact

            if expected_fail_cid in TARGET_CONTACTS
                ssd_fail_by_expected_contact[expected_fail_cid] += 1
            end

            fail_status = "ssd_simulation_failed"

            if is_outside_semiconductor_error(err)
                n_outside_semiconductor_fail += 1
                fail_status = "outside_semiconductor_failed"

                if expected_fail_cid in TARGET_CONTACTS
                    outside_semiconductor_fail_by_expected_contact[expected_fail_cid] += 1
                end
            else
                n_other_ssd_fail += 1
            end

            @warn "SSD simulation failed for Geant4 event $geant_event_id, expected contact $expected_fail_cid" exception = (err, catch_backtrace())

            append_summary!(
                summary_csv,
                geant_event_id,
                total_geant_edep,
                length(hits),
                fail_status,
                expected_fail_cid,
                geant_second_contact,
                NaN,
                NaN,
                NaN,
                NaN,
                NaN,
                expected_fail_cid
            )
        end
    end

    # Build averaged superpulses.
    superpulses = Dict{Int, Matrix{Float32}}()

    for cid in TARGET_CONTACTS
        if counts[cid] > 0
            superpulses[cid] = accum[cid] ./ Float32(counts[cid])

            contact_file = joinpath(OUT_DIR, "$(OUT_PREFIX)_contact$(cid).jls")
            serialize(contact_file, superpulses[cid])

            println()
            println("Saved superpulse for contact ", cid)
            println("  count = ", counts[cid])
            println("  file  = ", contact_file)

            plot_superpulse(superpulses[cid], cid)

            append_superpulse_amplitude_summary!(
                OUT_AMP_SUMMARY_CSV,
                cid,
                superpulses[cid]
            )
        else
            println()
            println("No accepted events for contact ", cid)
        end
    end

    output = Dict(
        "superpulses" => superpulses,
        "counts" => counts,
        "time_ns" => Float32.(ML_TIME_NS),
        "contact_ids" => Int32.(ALL_CONTACT_IDS),
        "metadata" => Dict(
            "simulation_file" => SIM_FILE,
            "geant_hits_file" => GEANT_HITS_FILE,
            "phi110_deg" => Float32(PHI110_DEG),
            "temperature_K" => Float32(TEMPERATURE_K),
            "drift_dt_ns" => Float32(ustrip(u"ns", DRIFT_DT)),
            "ml_dt_ns" => Float32(ML_DT_NS),
            "ml_trace_length_ns" => Float32(ML_TRACE_LENGTH_NS),
            "nsamples" => Int32(NSAMPLES),
            "ncontacts" => Int32(NCONTACTS),
            "target_contacts" => Int32.(TARGET_CONTACTS),
            "n_accepted_per_contact_requested" => Int32(N_ACCEPTED_PER_CONTACT),
            "apply_geant_energy_gate" => APPLY_GEANT_ENERGY_GATE,
            "photopeak_energy_keV" => Float32(PHOTOPEAK_ENERGY_KEV),
            "photopeak_tolerance_keV" => Float32(PHOTOPEAK_TOLERANCE_KEV),
            "apply_geant_single_contact_gate" => APPLY_GEANT_SINGLE_CONTACT_GATE,
            "min_geant_contact_fraction" => Float32(MIN_GEANT_CONTACT_FRACTION),
            "max_geant_second_contact_energy_keV" => Float32(MAX_GEANT_SECOND_CONTACT_ENERGY_KEV),
            "require_geant_ssd_contact_agreement" => REQUIRE_GEANT_SSD_CONTACT_AGREEMENT,
            "apply_segment_boundary_filter" => APPLY_SEGMENT_BOUNDARY_FILTER,
            "phi_boundary_margin_deg" => Float32(PHI_BOUNDARY_MARGIN_DEG),
            "z_boundary_margin_mm" => Float32(Z_BOUNDARY_MARGIN_MM),
            "apply_taper_boundary_filter" => APPLY_TAPER_BOUNDARY_FILTER,
            "taper_boundary_margin_mm" => Float32(TAPER_BOUNDARY_MARGIN_MM),
            "min_segment_confidence" => Float32(MIN_SEGMENT_CONFIDENCE),
            "trigger_alignment_applied" => APPLY_TRIGGER_ALIGNMENT,
            "trigger_contact_id" => Int32(TRIGGER_CONTACT_ID),
            "trigger_fraction" => Float32(TRIGGER_FRACTION),
            "trigger_time_ns" => Float32(TRIGGER_TIME_NS),
            "preamp_response_applied" => APPLY_PREAMP_RESPONSE,
            "integral_crosstalk_applied" => APPLY_INTEGRAL_CROSSTALK,
            "differential_crosstalk_applied" => APPLY_DIFFERENTIAL_CROSSTALK,
            "noise_added" => ADD_NOISE
        )
    )

    out_file = joinpath(OUT_DIR, "$(OUT_PREFIX).jls")
    serialize(out_file, output)

    diagnostic_csv = joinpath(OUT_DIR, "$(OUT_PREFIX)_contact_diagnostics.csv")
    write_contact_diagnostics!(
        diagnostic_csv,
        best_contact_after_energy_gate,
        geant_single_contact_fail_by_expected_contact,
        contact_mismatch_fail_by_expected_contact,
        confidence_fail_by_contact,
        target_final_fail_by_contact,
        core_final_fail_by_contact,
        target_already_full_by_contact,
        accepted_by_contact
    )

    failure_diagnostic_csv = joinpath(OUT_DIR, "$(OUT_PREFIX)_ssd_failure_diagnostics.csv")
    write_failure_diagnostics!(
        failure_diagnostic_csv,
        ssd_fail_by_expected_contact,
        outside_semiconductor_fail_by_expected_contact
    )

    println()
    println("============================================================")
    println("Done.")
    println("Processed Geant4 events:                 ", n_processed)
    println("Accepted total SSD events:               ", n_accepted_total)
    println("Energy gate failures:                    ", n_energy_gate_fail)
    println("Geant single-contact failures:           ", n_geant_single_contact_fail)
    println("Geant/SSD contact mismatch failures:     ", n_contact_mismatch_fail)
    println("Segment-boundary failures:               ", n_segment_boundary_fail)
    println("Taper-boundary failures:                 ", n_taper_boundary_fail)
    println("SSD failures:                            ", n_ssd_fail)
    println("Missing waveform failures:               ", n_missing_waveforms)
    println("Confidence failures:                     ", n_confidence_fail)
    println()
    println("Accepted counts by contact:")
    for cid in TARGET_CONTACTS
        println("  contact ", cid, ": ", counts[cid])
    end

    println()
    println("SSD failure breakdown:")
    println("  outside semiconductor failures: ", n_outside_semiconductor_fail)
    println("  other SSD failures:             ", n_other_ssd_fail)

    println()
    println("SSD failures by expected contact:")
    for cid in TARGET_CONTACTS
        println("  contact ", cid, ": ", ssd_fail_by_expected_contact[cid])
    end

    println()
    println("Outside-semiconductor failures by expected contact:")
    for cid in TARGET_CONTACTS
        println("  contact ", cid, ": ", outside_semiconductor_fail_by_expected_contact[cid])
    end

    println()
    println("Saved combined superpulse file:")
    println("  ", out_file)
    println("Saved summary CSV:")
    println("  ", summary_csv)
    println("Saved amplitude summary CSV:")
    println("  ", OUT_AMP_SUMMARY_CSV)
    println("Saved accepted-position CSV:")
    println("  ", OUT_ACCEPTED_POSITIONS_CSV)
    println("Saved contact diagnostics CSV:")
    println("  ", diagnostic_csv)
    println("Saved SSD failure diagnostics CSV:")
    println("  ", failure_diagnostic_csv)
    println("============================================================")
end

# ============================================================
# Geant4 input parsing
# ============================================================

function read_geant_hits(path)
    events = Dict{Int, Vector{NamedTuple}}()

    open(path, "r") do io
        header = readline(io)
        cols = split(header, ",")

        col_index = Dict(strip(c) => i for (i, c) in enumerate(cols))

        required = ["event_id", "track_id", "x_mm", "y_mm", "z_ssd_mm", "edep_keV"]

        for name in required
            if !haskey(col_index, name)
                error("Missing required column '$name' in $path. Found columns: $cols")
            end
        end

        for line in eachline(io)
            isempty(strip(line)) && continue

            parts = split(line, ",")

            event_id = parse(Int, strip(parts[col_index["event_id"]]))
            track_id = parse(Int, strip(parts[col_index["track_id"]]))
            x_mm = parse(Float64, strip(parts[col_index["x_mm"]]))
            y_mm = parse(Float64, strip(parts[col_index["y_mm"]]))
            z_ssd_mm = parse(Float64, strip(parts[col_index["z_ssd_mm"]]))
            edep_keV = parse(Float64, strip(parts[col_index["edep_keV"]]))

            if edep_keV < MIN_HIT_EDEP_KEV
                continue
            end

            hit = (
                event_id = event_id,
                track_id = track_id,
                x_mm = x_mm,
                y_mm = y_mm,
                z_mm = z_ssd_mm,
                edep_keV = edep_keV
            )

            if !haskey(events, event_id)
                events[event_id] = NamedTuple[]
            end

            push!(events[event_id], hit)
        end
    end

    return events
end

# ============================================================
# Geant4-to-SSD event conversion
# ============================================================

function make_ssd_event_from_hits(T, hits)
    starting_positions = CartesianPoint{T}[]
    energy_values = T[]

    for h in hits
        push!(
            starting_positions,
            CartesianPoint{T}(
                T(h.x_mm * 1e-3),
                T(h.y_mm * 1e-3),
                T(h.z_mm * 1e-3)
            )
        )

        push!(energy_values, T(h.edep_keV))
    end

    return Event(starting_positions, energy_values * u"keV")
end

# ============================================================
# Expected contact from Geant4 coordinates
# ============================================================

function phi_deg_from_xy(x_mm, y_mm)
    phi = atan(y_mm, x_mm) * 180.0 / pi

    if phi < 0
        phi += 360.0
    end

    return phi
end

function expected_contact_from_position(x_mm, y_mm, z_mm)
    phi = phi_deg_from_xy(x_mm, y_mm)

    if z_mm >= 70.0
        if 90.0 <= phi < 180.0
            return 1
        elseif 180.0 <= phi < 270.0
            return 2
        elseif 270.0 <= phi < 360.0
            return 3
        else
            return 4
        end
    else
        if 90.0 <= phi < 180.0
            return 5
        elseif 180.0 <= phi < 270.0
            return 6
        elseif 270.0 <= phi < 360.0
            return 7
        else
            return 8
        end
    end
end

function expected_contact_from_largest_hit(hits)
    edeps = [h.edep_keV for h in hits]
    h = hits[argmax(edeps)]

    return expected_contact_from_position(h.x_mm, h.y_mm, h.z_mm)
end

function geant_contact_energy_summary(hits)
    e_by_contact = Dict(cid => 0.0 for cid in TARGET_CONTACTS)

    for h in hits
        cid = expected_contact_from_position(h.x_mm, h.y_mm, h.z_mm)

        if cid in TARGET_CONTACTS
            e_by_contact[cid] += h.edep_keV
        end
    end

    sorted_contacts = sort(
        collect(e_by_contact),
        by = x -> x[2],
        rev = true
    )

    best_contact = sorted_contacts[1][1]
    best_energy = sorted_contacts[1][2]

    second_contact = sorted_contacts[2][1]
    second_energy = sorted_contacts[2][2]

    total_energy = sum(values(e_by_contact))
    fraction = total_energy > 0 ? best_energy / total_energy : 0.0

    return best_contact, second_contact, best_energy, second_energy, fraction, e_by_contact
end

function energy_weighted_centroid(hits)
    etot = sum(h.edep_keV for h in hits)

    if etot <= 0
        return NaN, NaN, NaN
    end

    x = sum(h.edep_keV * h.x_mm for h in hits) / etot
    y = sum(h.edep_keV * h.y_mm for h in hits) / etot
    z = sum(h.edep_keV * h.z_mm for h in hits) / etot

    return x, y, z
end

# ============================================================
# Boundary helpers
# ============================================================

function angular_distance_to_nearest_segment_boundary_deg(phi)
    boundaries = [0.0, 90.0, 180.0, 270.0, 360.0]
    return minimum(abs.(phi .- boundaries))
end

function is_too_close_to_segment_boundary(
    h;
    phi_margin_deg = PHI_BOUNDARY_MARGIN_DEG,
    z_margin_mm = Z_BOUNDARY_MARGIN_MM
)
    phi = phi_deg_from_xy(h.x_mm, h.y_mm)

    dphi = angular_distance_to_nearest_segment_boundary_deg(phi)
    dz = abs(h.z_mm - 70.0)

    return dphi < phi_margin_deg || dz < z_margin_mm
end

function event_too_close_to_segment_boundary(hits)
    return any(h -> is_too_close_to_segment_boundary(h), hits)
end

function taper_margins_mm(x, y, z)
    if z < 60.0
        return (Inf, Inf)
    end

    oct_apothem = 29.0
    taper_start_z = 60.0
    taper_angle_deg = 22.5

    dz = z - taper_start_z
    t = tan(deg2rad(taper_angle_deg))

    left_plane_x = -oct_apothem + dz * t
    top_plane_y = oct_apothem - dz * t

    # Positive margin means safely inside with respect to taper planes.
    margin_left = x - left_plane_x
    margin_top = top_plane_y - y

    return (margin_left, margin_top)
end

function is_too_close_to_taper_boundary(h; margin_mm = 0.25)
    ml, mt = taper_margins_mm(h.x_mm, h.y_mm, h.z_mm)

    return ml < margin_mm || mt < margin_mm
end

function is_outside_semiconductor_error(err)
    msg = lowercase(sprint(showerror, err))

    return occursin("outside the semiconductor", msg) ||
           occursin("outside semiconductor", msg) ||
           occursin("electron cloud", msg)
end

# ============================================================
# Waveform extraction and resampling
# ============================================================

function extract_waveforms(evt, sim)
    signals = Dict{Int, Any}()

    contact_ids = [c.id for c in sim.detector.contacts]

    if !(:waveforms in propertynames(evt))
        error("Event has no :waveforms property.")
    end

    for i in eachindex(evt.waveforms)
        cid = contact_ids[i]
        wf = evt.waveforms[i]

        if wf !== missing
            signals[cid] = wf
        end
    end

    return signals
end

function signal_to_xy(wf)
    t_raw = getfield(wf, 1)
    y_raw = getfield(wf, 2)

    t_vec = collect(t_raw)
    y_vec = collect(y_raw)

    x_ns = vector_to_ns(t_vec)
    y_keV = vector_to_keV(y_vec)

    n = min(length(x_ns), length(y_keV))

    return x_ns[1:n], y_keV[1:n]
end

function vector_to_ns(v)
    try
        return Float64.(ustrip.(u"ns", v))
    catch
        return Float64.(v)
    end
end

function vector_to_keV(v)
    try
        return Float64.(ustrip.(u"keV", v))
    catch
        try
            return Float64.(ustrip.(v))
        catch
            return Float64.(v)
        end
    end
end

function build_fixed_waveform_matrix(signals)
    wf_matrix = zeros(Float32, NCONTACTS, NSAMPLES)

    if length(ML_TIME_NS) != NSAMPLES
        error("ML_TIME_NS length $(length(ML_TIME_NS)) does not match NSAMPLES $NSAMPLES")
    end

    for cid in ALL_CONTACT_IDS
        t_ns, amp_keV = signal_to_xy(signals[cid])
        y_interp = interp1_constant_edges(t_ns, amp_keV, ML_TIME_NS)

        if length(y_interp) != NSAMPLES
            y_fixed = zeros(Float64, NSAMPLES)
            ncopy = min(length(y_interp), NSAMPLES)
            y_fixed[1:ncopy] .= y_interp[1:ncopy]

            if length(y_interp) > 0 && ncopy < NSAMPLES
                y_fixed[ncopy+1:end] .= y_interp[end]
            end

            y_interp = y_fixed
        end

        idx = cid_to_index(cid)
        wf_matrix[idx, :] .= Float32.(y_interp)
    end

    return wf_matrix
end

function cid_to_index(cid)
    idx = findfirst(==(cid), ALL_CONTACT_IDS)

    idx === nothing && error("Unknown contact id: $cid")

    return idx
end

function interp1_constant_edges(x, y, xnew)
    n = length(x)

    if n == 0
        return zeros(length(xnew))
    elseif n == 1
        return fill(y[1], length(xnew))
    end

    out = similar(xnew, Float64)
    j = 1

    for i in eachindex(xnew)
        xx = xnew[i]

        if xx <= x[1]
            out[i] = y[1]
            continue
        end

        if xx >= x[end]
            out[i] = y[end]
            continue
        end

        while j < n - 1 && x[j + 1] < xx
            j += 1
        end

        x0 = x[j]
        x1 = x[j + 1]
        y0 = y[j]
        y1 = y[j + 1]

        if x1 == x0
            out[i] = y0
        else
            f = (xx - x0) / (x1 - x0)
            out[i] = y0 + f * (y1 - y0)
        end
    end

    return out
end

# ============================================================
# Baseline and trigger alignment
# ============================================================

function baseline_subtract_waveform_matrix!(
    wf_matrix;
    baseline_start_ns = 0.0,
    baseline_end_ns = 50.0
)
    baseline_idx = findall(t -> baseline_start_ns <= t <= baseline_end_ns, ML_TIME_NS)

    if isempty(baseline_idx)
        error("No samples found in baseline window.")
    end

    for cid in ALL_CONTACT_IDS
        idx = cid_to_index(cid)
        baseline = mean(wf_matrix[idx, baseline_idx])
        wf_matrix[idx, :] .-= baseline
    end

    return wf_matrix
end

function align_waveform_matrix_to_fraction_trigger(
    wf_matrix,
    trigger_cid;
    fraction = TRIGGER_FRACTION,
    trigger_time_ns = TRIGGER_TIME_NS
)
    trigger_idx = cid_to_index(trigger_cid)

    y = Float64.(wf_matrix[trigger_idx, :])

    y0 = y[1]
    yf = y[end]
    dy = yf - y0

    if abs(dy) <= 0
        return wf_matrix
    end

    threshold = y0 + fraction * dy

    if dy > 0
        crossing_idx = findfirst(v -> v >= threshold, y)
    else
        crossing_idx = findfirst(v -> v <= threshold, y)
    end

    if crossing_idx === nothing
        return wf_matrix
    end

    trigger_crossing_time = ML_TIME_NS[crossing_idx]
    shifted_time_axis = ML_TIME_NS .+ trigger_crossing_time .- trigger_time_ns

    aligned = zeros(Float32, size(wf_matrix))

    for cid in ALL_CONTACT_IDS
        idx = cid_to_index(cid)

        y_old = Float64.(wf_matrix[idx, :])
        y_new = interp1_constant_edges(ML_TIME_NS, y_old, shifted_time_axis)

        aligned[idx, :] .= Float32.(y_new)
    end

    return aligned
end

# ============================================================
# Optional electronics
# ============================================================

function apply_preamp_response!(wf_matrix)
    for cid in ALL_CONTACT_IDS
        idx = cid_to_index(cid)

        if cid == CORE_CONTACT_ID
            rise = CORE_PREAMP_RISE_10_90_NS
            decay = CORE_PREAMP_DECAY_TAU_NS
        else
            rise = OUTER_PREAMP_RISE_10_90_NS
            decay = OUTER_PREAMP_DECAY_TAU_NS
        end

        wf_matrix[idx, :] .= preamp_shape_charge_waveform(
            wf_matrix[idx, :],
            Float32(ML_DT_NS),
            Float32(rise),
            Float32(decay)
        )
    end

    return wf_matrix
end

function preamp_shape_charge_waveform(q_in, dt_ns::Float32, rise_10_90_ns::Float32, decay_tau_ns::Float32)
    n = length(q_in)

    q_rise = similar(q_in)
    y_out = similar(q_in)

    tau_rise_ns = rise_10_90_ns / Float32(log(9.0))

    if tau_rise_ns <= 0
        q_rise .= q_in
    else
        alpha_r = exp(-dt_ns / tau_rise_ns)
        q_rise[1] = q_in[1]

        for i in 2:n
            q_rise[i] = alpha_r * q_rise[i - 1] + (1.0f0 - alpha_r) * q_in[i]
        end
    end

    if decay_tau_ns <= 0
        y_out .= q_rise
    else
        alpha_d = exp(-dt_ns / decay_tau_ns)
        y_out[1] = q_rise[1]

        for i in 2:n
            y_out[i] = alpha_d * y_out[i - 1] + (q_rise[i] - q_rise[i - 1])
        end
    end

    return y_out
end

function apply_integral_crosstalk!(wf_matrix)
    mixed = copy(wf_matrix)

    for i in 1:NCONTACTS
        for j in 1:NCONTACTS
            if i != j
                mixed[i, :] .+= IXT_COEFF .* wf_matrix[j, :]
            end
        end
    end

    wf_matrix .= mixed
    return wf_matrix
end

function apply_differential_crosstalk!(wf_matrix)
    deriv = zeros(Float32, size(wf_matrix))

    for ch in 1:NCONTACTS
        if NSAMPLES >= 2
            deriv[ch, 1] = (wf_matrix[ch, 2] - wf_matrix[ch, 1]) / Float32(ML_DT_NS)

            for k in 2:NSAMPLES-1
                deriv[ch, k] = (wf_matrix[ch, k+1] - wf_matrix[ch, k-1]) / Float32(2.0 * ML_DT_NS)
            end

            deriv[ch, NSAMPLES] = (wf_matrix[ch, NSAMPLES] - wf_matrix[ch, NSAMPLES - 1]) / Float32(ML_DT_NS)
        end
    end

    mixed = copy(wf_matrix)

    for i in 1:NCONTACTS
        for j in 1:NCONTACTS
            if i != j
                mixed[i, :] .+= DXT_COEFF_NS .* deriv[j, :]
            end
        end
    end

    wf_matrix .= mixed
    return wf_matrix
end

# ============================================================
# Segment identification
# ============================================================

function final_amplitudes(signals)
    finals = Dict{Int, Float64}()

    for cid in keys(signals)
        _, amp = signal_to_xy(signals[cid])

        # Use final-minus-initial because initial induced signal can be nonzero.
        finals[cid] = amp[end] - amp[1]
    end

    return finals
end

function identify_collecting_segment(finals)
    outer_vals = [(cid, finals[cid]) for cid in OUTER_CONTACT_IDS]

    # Collecting contact should have the largest absolute final collected charge.
    sorted_outer = sort(
        outer_vals,
        by = x -> abs(x[2]),
        rev = true
    )

    best_cid = sorted_outer[1][1]
    best_val = sorted_outer[1][2]

    second_cid = sorted_outer[2][1]
    second_val = sorted_outer[2][2]

    denom = abs(best_val) > 0 ? abs(best_val) : 1.0
    confidence = (abs(best_val) - abs(second_val)) / denom

    return best_cid, second_cid, best_val, second_val, confidence
end

# ============================================================
# CSV logging and checks
# ============================================================

function append_summary!(
    path,
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
    expected_contact
)
    open(path, "a") do io
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
                    expected_contact
                ),
                ","
            )
        )
    end
end

function append_accepted_position!(
    path,
    geant_event_id,
    best_cid,
    total_geant_edep,
    hits,
    geant_contact_fraction,
    geant_second_contact,
    geant_second_energy
)
    xbar, ybar, zbar = energy_weighted_centroid(hits)
    rbar = sqrt(xbar^2 + ybar^2)
    phibar = phi_deg_from_xy(xbar, ybar)

    open(path, "a") do io
        println(
            io,
            join(
                (
                    geant_event_id,
                    best_cid,
                    total_geant_edep,
                    xbar,
                    ybar,
                    zbar,
                    rbar,
                    phibar,
                    length(hits),
                    geant_contact_fraction,
                    geant_second_contact,
                    geant_second_energy
                ),
                ","
            )
        )
    end
end

function check_contacts(sim)
    contact_ids = sort([c.id for c in sim.detector.contacts])

    println()
    println("Detector contact IDs:")
    println("  ", contact_ids)

    for cid in ALL_CONTACT_IDS
        if !(cid in contact_ids)
            error("Expected contact id $cid, but detector only has contacts $contact_ids")
        end
    end

    println("All expected contacts are present.")
end

function write_contact_diagnostics!(
    path,
    best_contact_after_energy_gate,
    geant_single_contact_fail_by_expected_contact,
    contact_mismatch_fail_by_expected_contact,
    confidence_fail_by_contact,
    target_final_fail_by_contact,
    core_final_fail_by_contact,
    target_already_full_by_contact,
    accepted_by_contact
)
    open(path, "w") do io
        println(
            io,
            "contact,best_after_energy_gate,geant_single_contact_fail,geant_ssd_contact_mismatch,confidence_fail,target_final_fail,core_final_fail,target_already_full,accepted"
        )

        for cid in TARGET_CONTACTS
            println(
                io,
                join(
                    (
                        cid,
                        best_contact_after_energy_gate[cid],
                        geant_single_contact_fail_by_expected_contact[cid],
                        contact_mismatch_fail_by_expected_contact[cid],
                        confidence_fail_by_contact[cid],
                        target_final_fail_by_contact[cid],
                        core_final_fail_by_contact[cid],
                        target_already_full_by_contact[cid],
                        accepted_by_contact[cid]
                    ),
                    ","
                )
            )
        end
    end
end

function write_failure_diagnostics!(
    path,
    ssd_fail_by_expected_contact,
    outside_semiconductor_fail_by_expected_contact
)
    open(path, "w") do io
        println(io, "contact,ssd_failures,outside_semiconductor_failures")

        for cid in TARGET_CONTACTS
            println(
                io,
                join(
                    (
                        cid,
                        ssd_fail_by_expected_contact[cid],
                        outside_semiconductor_fail_by_expected_contact[cid]
                    ),
                    ","
                )
            )
        end
    end
end

# ============================================================
# Plotting and amplitude summaries
# ============================================================

function plot_superpulse(superpulse, target_cid)
    p = plot(
        title = "SSD simulated superpulse, collecting contact $(target_cid)",
        xlabel = "time / ns",
        ylabel = "signal / keV",
        size = (1100, 750),
        legend = :outerright,
        xlims = (0, 700)
    )

    for cid in ALL_CONTACT_IDS
        idx = cid_to_index(cid)

        plot!(
            p,
            ML_TIME_NS,
            superpulse[idx, :],
            label = "contact $cid",
            linewidth = cid == target_cid ? 3.0 : cid == CORE_CONTACT_ID ? 2.5 : 1.3
        )
    end

    out_png = joinpath(OUT_DIR, "$(OUT_PREFIX)_contact$(target_cid).png")
    savefig(p, out_png)

    println("Saved plot:")
    println("  ", out_png)
end

function write_event_energy_track_summary!(events, csv_path)
    open(csv_path, "w") do io
        println(io, "event_id,total_edep_keV,total_tracks,n_hits,largest_hit_expected_contact,geant_best_contact,geant_second_contact,geant_contact_fraction,geant_second_contact_energy_keV")

        for event_id in sort(collect(keys(events)))
            hits = events[event_id]

            total_edep = sum(h.edep_keV for h in hits)
            total_tracks = length(unique([h.track_id for h in hits]))
            n_hits = length(hits)

            largest_hit_contact = expected_contact_from_largest_hit(hits)

            geant_best_contact,
            geant_second_contact,
            geant_best_energy,
            geant_second_energy,
            geant_contact_fraction,
            geant_e_by_contact = geant_contact_energy_summary(hits)

            println(
                io,
                join(
                    (
                        event_id,
                        total_edep,
                        total_tracks,
                        n_hits,
                        largest_hit_contact,
                        geant_best_contact,
                        geant_second_contact,
                        geant_contact_fraction,
                        geant_second_energy
                    ),
                    ","
                )
            )
        end
    end
end

function plot_event_energy_track_bars(events, out_png; max_events = MAX_EVENTS_IN_EVENT_BAR_PLOT)
    event_ids_all = sort(collect(keys(events)))

    if isempty(event_ids_all)
        println("No events available for event energy/track bar plot.")
        return
    end

    event_ids = event_ids_all[1:min(max_events, length(event_ids_all))]

    total_edep_keV = Float64[]
    total_tracks = Float64[]

    for event_id in event_ids
        hits = events[event_id]

        push!(total_edep_keV, sum(h.edep_keV for h in hits))
        push!(total_tracks, length(unique([h.track_id for h in hits])))
    end

    energy_norm = total_edep_keV ./ PHOTOPEAK_ENERGY_KEV

    max_tracks = maximum(total_tracks)
    track_norm = max_tracks > 0 ? total_tracks ./ max_tracks : total_tracks

    n = length(event_ids)
    x = collect(1:n)

    bar_width = 0.38

    p = plot(
        title = "Geant4 Event Energy and Track Count",
        xlabel = "event_id",
        ylabel = "normalized value",
        size = (1400, 700),
        legend = :topright,
        xticks = (x, string.(event_ids)),
        xrotation = 60
    )

    bar!(
        p,
        x .- bar_width / 2,
        energy_norm,
        bar_width = bar_width,
        label = "total energy / $(PHOTOPEAK_ENERGY_KEV) keV"
    )

    bar!(
        p,
        x .+ bar_width / 2,
        track_norm,
        bar_width = bar_width,
        label = "tracks / max tracks"
    )

    savefig(p, out_png)

    println("Saved event energy/track plot:")
    println("  ", out_png)
end

function signed_peak_amplitude(y)
    idx = argmax(abs.(y))
    return Float64(y[idx])
end

function final_amplitude(y)
    return Float64(y[end])
end

function append_superpulse_amplitude_summary!(csv_path, target_cid, superpulse)
    target_idx = cid_to_index(target_cid)
    core_idx = cid_to_index(CORE_CONTACT_ID)

    target_peak = signed_peak_amplitude(superpulse[target_idx, :])
    target_final = final_amplitude(superpulse[target_idx, :])

    core_peak = signed_peak_amplitude(superpulse[core_idx, :])
    core_final = final_amplitude(superpulse[core_idx, :])

    outer_values = [
        (
            cid,
            signed_peak_amplitude(superpulse[cid_to_index(cid), :]),
            final_amplitude(superpulse[cid_to_index(cid), :])
        )
        for cid in OUTER_CONTACT_IDS
        if cid != target_cid
    ]

    sorted_outer = sort(
        outer_values,
        by = x -> abs(x[2]),
        rev = true
    )

    second_cid = sorted_outer[1][1]
    second_peak = sorted_outer[1][2]
    second_final = sorted_outer[1][3]

    second_peak_over_target_peak =
        second_peak / (abs(target_peak) > 1e-9 ? target_peak : 1e-9)

    second_final_over_target_final =
        second_final / (abs(target_final) > 1e-9 ? target_final : 1e-9)

    open(csv_path, "a") do io
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
                    second_peak_over_target_peak,
                    second_final_over_target_final
                ),
                ","
            )
        )
    end
end

# ============================================================
# Run
# ============================================================

main()