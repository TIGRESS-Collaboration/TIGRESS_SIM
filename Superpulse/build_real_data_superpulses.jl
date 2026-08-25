# The numeric field between the contact label and ':' is parsed only as
# syntax and is NEVER stored or used. All target selection, calibration,
# energy gating, confidence, averaging, plots, and CSV values come from
# waveform samples.

using Serialization
using Statistics
using Plots

ENV["GKSwstype"] = "100"
gr()
closeall()
default(background_color=:white, background_color_inside=:white,
        foreground_color=:black, dpi=150)

# ============================ Settings ============================
const IN_WAVEFORM_FILE = "/data1/flerner/hpge_sims/CNN/realdata/out.dat"
const OUT_DIR = "/data1/flerner/hpge_sims/CNN/real_waveform_only_superpulses"
const OUT_PREFIX = "real_waveform_only_superpulses"
const SUMMARY_CSV = joinpath(OUT_DIR, "$(OUT_PREFIX)_event_summary.csv")
const AMP_SUMMARY_CSV = joinpath(OUT_DIR, "$(OUT_PREFIX)_amplitude_summary.csv")
const PARSER_SUMMARY_CSV = joinpath(OUT_DIR, "$(OUT_PREFIX)_parser_summary.csv")

const NSAMPLES = 100
const OUTER_CONTACT_IDS = collect(1:8)
const CORE_CONTACT_ID = 9
const ALL_CONTACT_IDS = collect(1:9)
const NCONTACTS = 9

const BASELINE_START_SAMPLE = 1
const BASELINE_END_SAMPLE = 30
const FINAL_START_SAMPLE = 90
const FINAL_END_SAMPLE = 99

const SAMPLE_PERIOD_NS = 10.0
const TIME_AXIS_NS = collect(0:NSAMPLES-1) .* SAMPLE_PERIOD_NS
const PLOT_XLIMS_NS = (TIME_AXIS_NS[1], TIME_AXIS_NS[end])
const PLOT_YLIMS_KEV = nothing

const PHOTOPEAK_ENERGY_KEV = 1332.5
const APPLY_1332_WAVEFORM_GATE = true
const ENERGY_GATE_HALF_WIDTH_KEV = 10.0
const MIN_TARGET_PEAK_KEV = 0.0
const APPLY_CONFIDENCE_CUT = true
const MIN_FINAL_CONFIDENCE = 0.98
const MAX_EVENTS_TO_PROCESS = typemax(Int)
const PRINT_PROGRESS_EVERY = 1000
const MAX_INCOMPLETE_WARNINGS = 50
const FLIP_CORE_POLARITY = false

# found using values before the : in dataset
"const WAVEFORM_1332_ADC = Dict(
    1 => 1004.8727477419355,
    2 => 1360.0,
    3 => 1019.2663061403509,
    4 => 994.6761642276423,
    5 => 1022.0220822085892,
    6 => 997.0248446428573,
    7 => 1000.0343905172413,
    8 => 1016.7803192660549,
    9 => 1038.8357202734373,
)"
const WAVEFORM_1332_ADC = Dict(
    1 => 980.3978243978245,
    2 => 1329.093149540518,
    3 => 986.625641025641,
    4 => 972.1213696369637,
    5 => 995.7892176199867,
    6 => 974.9032280701754,
    7 => 978.573590504451,
    8 => 990.8224089635855,
    9 => 1012.8978067318131
)
const CONTACT_ADC_TO_KEV = Dict(
    cid => PHOTOPEAK_ENERGY_KEV / WAVEFORM_1332_ADC[cid]
    for cid in ALL_CONTACT_IDS
)

# ============================ Sample waveform output ============================
const SAVE_SAMPLE_WAVEFORMS = true

# Events must be assigned to this target contact after target selection.
# Example: 3 means "events whose target contact is 3"
const SAMPLE_TARGET_CONTACT = 7

# Output this contact's waveform from those selected events.
# Example: 7 means "save contact 7 waveform for events targeting contact 3"
const SAMPLE_OUTPUT_CONTACT = 3

# Maximum number of accepted sample waveforms to save/plot
const MAX_SAMPLE_WAVEFORMS = 25

const SAMPLE_WAVEFORM_CSV = joinpath(
    OUT_DIR,
    "$(OUT_PREFIX)_target$(SAMPLE_TARGET_CONTACT)_contact$(SAMPLE_OUTPUT_CONTACT)_sample_waveforms.csv"
)

const SAMPLE_WAVEFORM_PNG = joinpath(
    OUT_DIR,
    "$(OUT_PREFIX)_target$(SAMPLE_TARGET_CONTACT)_contact$(SAMPLE_OUTPUT_CONTACT)_sample_waveforms.png"
)


# ============================ Main ============================
function main()
    mkpath(OUT_DIR)
    validate_settings()
    events, stats = read_real_waveform_events(IN_WAVEFORM_FILE)
    isempty(events) && error("No complete events parsed from $IN_WAVEFORM_FILE")
    write_parser_summary(PARSER_SUMMARY_CSV, stats)

    accum = Dict(cid => zeros(Float64, NCONTACTS, NSAMPLES) for cid in OUTER_CONTACT_IDS)
    counts = Dict(cid => 0 for cid in OUTER_CONTACT_IDS)
    selected = Dict(cid => 0 for cid in OUTER_CONTACT_IDS)
    energy_fail = Dict(cid => 0 for cid in OUTER_CONTACT_IDS)
    confidence_fail = Dict(cid => 0 for cid in OUTER_CONTACT_IDS)
    amplitude_fail = Dict(cid => 0 for cid in OUTER_CONTACT_IDS)
    sample_waveforms = Vector{Vector{Float64}}()
    sample_event_indices = Int[]

    initialize_csvs()
    nprocessed = 0
    naccepted = 0

    for (event_index, raw_matrix) in enumerate(events)
        nprocessed += 1
        nprocessed > MAX_EVENTS_TO_PROCESS && break

        # Everything below is based only on waveform samples.
        wf = copy(raw_matrix)
        baseline_subtract_matrix!(wf)
        convert_waveform_matrix_to_keV!(wf)

        if FLIP_CORE_POLARITY
            wf[cid_to_index(CORE_CONTACT_ID), :] .*= -1
        end

        deltas = final_deltas(wf)

        target_cid, final_second_cid_from_selection, target_final_for_selection, final_second_for_selection =
            identify_contacts_from_final_values(wf)
        selected[target_cid] += 1

        target_peak = signed_peak(wf[cid_to_index(target_cid), :])

        transient_vals = [(cid, signed_peak(wf[cid_to_index(cid), :])) for cid in OUTER_CONTACT_IDS if cid != target_cid]
        sort!(transient_vals, by=x->abs(x[2]), rev=true)
        transient_second_cid, transient_second_peak = transient_vals[1]

        final_confidence, final_second_cid, final_second =
            final_charge_confidence(deltas, target_cid)

        target_final = deltas[target_cid]
        core_final = deltas[CORE_CONTACT_ID]
        waveform_energy = abs(target_final)
        status = "accepted"

        if waveform_energy < MIN_TARGET_PEAK_KEV
            status = "low_amplitude_reject"
            amplitude_fail[target_cid] += 1
        elseif APPLY_1332_WAVEFORM_GATE &&
               abs(waveform_energy - PHOTOPEAK_ENERGY_KEV) > ENERGY_GATE_HALF_WIDTH_KEV
            status = "energy_1332_reject"
            energy_fail[target_cid] += 1
        elseif APPLY_CONFIDENCE_CUT && final_confidence < MIN_FINAL_CONFIDENCE
            status = "confidence_reject"
            confidence_fail[target_cid] += 1
        else
            accum[target_cid] .+= wf
            counts[target_cid] += 1
            naccepted += 1

            if SAVE_SAMPLE_WAVEFORMS &&
            target_cid == SAMPLE_TARGET_CONTACT &&
            length(sample_waveforms) < MAX_SAMPLE_WAVEFORMS

                out_i = cid_to_index(SAMPLE_OUTPUT_CONTACT)
                push!(sample_waveforms, copy(wf[out_i, :]))
                push!(sample_event_indices, event_index)
            end
        end

        append_event_summary(SUMMARY_CSV, event_index, status,
            target_cid, transient_second_cid, final_second_cid,
            target_peak, transient_second_peak, target_final, final_second,
            core_final, waveform_energy, final_confidence)

        if nprocessed % PRINT_PROGRESS_EVERY == 0
            println("processed=$nprocessed accepted=$naccepted counts=$counts")
        end
    end

    superpulses = Dict{Int,Matrix{Float64}}()
    for cid in OUTER_CONTACT_IDS
        counts[cid] == 0 && continue
        sp = accum[cid] ./ counts[cid]
        superpulses[cid] = sp
        serialize(joinpath(OUT_DIR, "$(OUT_PREFIX)_contact$(cid).jls"), sp)
        plot_superpulse(sp, cid)
        append_amplitude_summary(AMP_SUMMARY_CSV, cid, sp)
        if SAVE_SAMPLE_WAVEFORMS
            write_sample_waveforms(SAMPLE_WAVEFORM_CSV, sample_waveforms, sample_event_indices)
            plot_sample_waveforms(SAMPLE_WAVEFORM_PNG, sample_waveforms, sample_event_indices)
        end
    end

    output = Dict(
        "superpulses" => superpulses,
        "counts" => counts,
        "time_ns" => TIME_AXIS_NS,
        "contact_ids" => ALL_CONTACT_IDS,
        "metadata" => Dict(
            "input_file" => IN_WAVEFORM_FILE,
            "ignored_header_peak_field" => true,
            "selection_source" => "waveform samples only",
            "target_selection" => "largest absolute signed waveform peak",
            "energy_selection" => "absolute target waveform peak",
            "confidence_selection" => "final waveform amplitudes",
            "waveform_output_unit" => "keV",
            "sample_period_ns" => SAMPLE_PERIOD_NS,
            "waveform_1332_adc" => WAVEFORM_1332_ADC,
            "contact_adc_to_kev" => CONTACT_ADC_TO_KEV,
            "energy_gate_half_width_keV" => ENERGY_GATE_HALF_WIDTH_KEV,
        )
    )
    serialize(joinpath(OUT_DIR, "$(OUT_PREFIX).jls"), output)

    println("\nAccepted counts by contact:")
    for cid in OUTER_CONTACT_IDS
        println("  contact $cid: $(counts[cid])")
    end
    println("\nSelection diagnostics:")
    for cid in OUTER_CONTACT_IDS
        println("  contact $cid: selected=$(selected[cid]), accepted=$(counts[cid]), " *
                "energy_fail=$(energy_fail[cid]), amplitude_fail=$(amplitude_fail[cid]), " *
                "confidence_fail=$(confidence_fail[cid])")
    end
    print_parser_stats(stats)
end

# ============================ Parsing ============================
# Matches and ignores the numeric field before ':'.
function parse_waveform_line(line)
    m = match(r"^\s*([A-Za-z0-9_]+)\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*:\s*(.*)$", strip(line))
    m === nothing && return nothing
    label = strip(m.captures[1])
    payload = strip(m.captures[2])
    samples = isempty(payload) ? Float64[] : parse.(Float64, strip.(split(payload, ",")))
    return label, samples
end

label_to_cid(label) = uppercase(strip(label)) == "FV" ? CORE_CONTACT_ID : parse(Int, strip(label))

function fix_waveform_length(values, stats)
    n = length(values)
    stats["n_waveform_lines_total"] += 1
    stats["min_original_length"] = min(stats["min_original_length"], n)
    stats["max_original_length"] = max(stats["max_original_length"], n)
    stats["sum_original_length"] += n
    if n == NSAMPLES
        stats["n_exact_length"] += 1
        return Float64.(values)
    elseif n > NSAMPLES
        stats["n_truncated"] += 1
        return Float64.(values[1:NSAMPLES])
    elseif n > 0
        stats["n_padded"] += 1
        return vcat(Float64.(values), fill(Float64(values[end]), NSAMPLES-n))
    else
        stats["n_empty_waveforms"] += 1
        return zeros(Float64, NSAMPLES)
    end
end

is_complete(d) = all(haskey(d,cid) && length(d[cid]) == NSAMPLES for cid in ALL_CONTACT_IDS)

function dict_to_matrix(d)
    m = zeros(Float64, NCONTACTS, NSAMPLES)
    for cid in ALL_CONTACT_IDS
        m[cid_to_index(cid), :] .= d[cid]
    end
    m
end

function init_stats()
    Dict{String,Any}(
        "n_lines_total"=>0, "n_waveform_lines_total"=>0, "n_unparsed_lines"=>0,
        "n_unknown_labels"=>0, "n_complete_events"=>0,
        "n_incomplete_blank_flushes"=>0, "n_duplicate_channel_flushes"=>0,
        "n_exact_length"=>0, "n_truncated"=>0, "n_padded"=>0,
        "n_empty_waveforms"=>0, "min_original_length"=>typemax(Int),
        "max_original_length"=>0, "sum_original_length"=>0,
        "n_incomplete_warnings_printed"=>0)
end

function flush_event!(events, current, stats, reason)
    isempty(current) && return
    if is_complete(current)
        push!(events, dict_to_matrix(current))
        stats["n_complete_events"] += 1
    else
        key = reason == "duplicate" ? "n_duplicate_channel_flushes" : "n_incomplete_blank_flushes"
        stats[key] += 1
        if stats["n_incomplete_warnings_printed"] < MAX_INCOMPLETE_WARNINGS
            missing = [cid for cid in ALL_CONTACT_IDS if !haskey(current,cid)]
            println("Skipped incomplete event ($reason), missing=$missing")
            stats["n_incomplete_warnings_printed"] += 1
        end
    end
    empty!(current)
end

function read_real_waveform_events(path)
    events = Matrix{Float64}[]
    current = Dict{Int,Vector{Float64}}()
    stats = init_stats()
    open(path, "r") do io
        for line in eachline(io)
            stats["n_lines_total"] += 1
            if isempty(strip(line))
                flush_event!(events,current,stats,"blank")
                continue
            end
            parsed = parse_waveform_line(line)
            if parsed === nothing
                stats["n_unparsed_lines"] += 1
                continue
            end
            label, values = parsed
            cid = try label_to_cid(label) catch; stats["n_unknown_labels"] += 1; continue end
            if !(cid in ALL_CONTACT_IDS)
                stats["n_unknown_labels"] += 1
                continue
            end
            fixed = fix_waveform_length(values, stats)
            haskey(current,cid) && flush_event!(events,current,stats,"duplicate")
            current[cid] = fixed
            if is_complete(current)
                push!(events, dict_to_matrix(current))
                stats["n_complete_events"] += 1
                empty!(current)
            end
        end
    end
    flush_event!(events,current,stats,"eof")
    return events, stats
end

# ============================ Waveforms ============================
cid_to_index(cid) = findfirst(==(cid), ALL_CONTACT_IDS)

function baseline_subtract_matrix!(wf)
    r = BASELINE_START_SAMPLE:BASELINE_END_SAMPLE
    for cid in ALL_CONTACT_IDS
        i = cid_to_index(cid)
        wf[i,:] .-= mean(wf[i,r])
    end
    wf
end

function convert_waveform_matrix_to_keV!(wf)
    for cid in ALL_CONTACT_IDS
        wf[cid_to_index(cid),:] .*= CONTACT_ADC_TO_KEV[cid]
    end
    wf
end

signed_peak(y) = Float64(y[argmax(abs.(y))])
final_level(y) = Float64(mean(y[FINAL_START_SAMPLE:FINAL_END_SAMPLE]))

function final_deltas(wf)
    Dict(cid => final_level(wf[cid_to_index(cid),:]) for cid in ALL_CONTACT_IDS)
end

function identify_contacts_from_final_values(wf)
    vals = [(cid, final_level(wf[cid_to_index(cid),:])) for cid in OUTER_CONTACT_IDS]
    sort!(vals, by=x->abs(x[2]), rev=true)
    return vals[1][1], vals[2][1], vals[1][2], vals[2][2]
end

function final_charge_confidence(deltas, target_cid)
    others = [(cid,deltas[cid]) for cid in OUTER_CONTACT_IDS if cid != target_cid]
    sort!(others, by=x->abs(x[2]), rev=true)
    second_cid, second_val = others[1]
    denom = max(abs(deltas[target_cid]), 1e-9)
    confidence = (abs(deltas[target_cid]) - abs(second_val)) / denom
    return confidence, second_cid, second_val
end

function identify_contacts_from_final_values(wf)
    vals = [(cid, final_level(wf[cid_to_index(cid),:])) for cid in OUTER_CONTACT_IDS]
    sort!(vals, by=x->abs(x[2]), rev=true)
    return vals[1][1], vals[2][1], vals[1][2], vals[2][2]
end

# ============================ CSV and plots ============================
function initialize_csvs()
    open(SUMMARY_CSV,"w") do io
        println(io,"event_index,status,target_contact,transient_second_contact,final_second_contact," *
                   "target_peak_keV,transient_second_peak_keV,target_final_keV,final_second_keV," *
                   "core_final_keV,waveform_energy_keV,final_confidence")
    end
    open(AMP_SUMMARY_CSV,"w") do io
        println(io,"target_contact,target_peak_keV,target_final_keV,core_contact,core_peak_keV," *
                   "core_final_keV,second_contact,second_peak_keV,second_final_keV," *
                   "second_peak_over_target_peak,second_final_over_target_final")
    end
end

function append_event_summary(path, args...)
    open(path,"a") do io
        println(io, join(args,","))
    end
end

function append_amplitude_summary(path,target_cid,sp)
    ti = cid_to_index(target_cid); ci = cid_to_index(CORE_CONTACT_ID)
    tp = signed_peak(sp[ti,:]); tf = final_level(sp[ti,:])
    cp = signed_peak(sp[ci,:]); cf = final_level(sp[ci,:])
    others = [(cid,signed_peak(sp[cid_to_index(cid),:]),final_level(sp[cid_to_index(cid),:]))
              for cid in OUTER_CONTACT_IDS if cid != target_cid]
    sort!(others,by=x->abs(x[2]),rev=true)
    sc,spk,sf = others[1]
    open(path,"a") do io
        println(io,join((target_cid,tp,tf,CORE_CONTACT_ID,cp,cf,sc,spk,sf,
                         spk/max(abs(tp),1e-9),sf/max(abs(tf),1e-9)),","))
    end
end

function plot_superpulse(sp,target_cid)
    p = plot(title="Waveform-only real superpulse, contact $target_cid",
             xlabel="time / ns",ylabel="baseline-subtracted signal / keV",
             size=(1100,750),legend=:outerright,xlims=PLOT_XLIMS_NS)
    for cid in ALL_CONTACT_IDS
        plot!(p,TIME_AXIS_NS,sp[cid_to_index(cid),:],label=cid==9 ? "FV/core" : "contact $cid",
              linewidth=cid==target_cid ? 3.0 : cid==9 ? 2.5 : 1.3)
    end
    PLOT_YLIMS_KEV !== nothing && ylims!(p,PLOT_YLIMS_KEV...)
    savefig(p,joinpath(OUT_DIR,"$(OUT_PREFIX)_contact$(target_cid).png"))
end

function write_parser_summary(path,stats)
    open(path,"w") do io
        println(io,"metric,value")
        for key in sort(collect(keys(stats)))
            key == "n_incomplete_warnings_printed" && continue
            value = key == "min_original_length" && stats[key] == typemax(Int) ? 0 : stats[key]
            println(io,"$key,$value")
        end
        n=stats["n_waveform_lines_total"]
        println(io,"avg_original_length,$(n>0 ? stats["sum_original_length"]/n : 0.0)")
    end
end

function print_parser_stats(stats)
    n=stats["n_waveform_lines_total"]
    avg=n>0 ? stats["sum_original_length"]/n : 0.0
    println("\nParser: complete=$(stats["n_complete_events"]), exact=$(stats["n_exact_length"]), " *
            "truncated=$(stats["n_truncated"]), padded=$(stats["n_padded"]), mean_length=$avg")
end

function validate_settings()
    1 <= BASELINE_START_SAMPLE <= BASELINE_END_SAMPLE <= NSAMPLES || error("Invalid baseline window")
    1 <= FINAL_START_SAMPLE <= FINAL_END_SAMPLE <= NSAMPLES || error("Invalid final window")
    for cid in ALL_CONTACT_IDS
        haskey(WAVEFORM_1332_ADC,cid) || error("Missing waveform calibration for contact $cid")
        WAVEFORM_1332_ADC[cid] > 0 || error("Invalid waveform calibration for contact $cid")
    end
end

function write_sample_waveforms(path, sample_waveforms, sample_event_indices)
    open(path, "w") do io
        header = ["event_index"]
        append!(header, ["sample_$(i)" for i in 1:NSAMPLES])
        println(io, join(header, ","))

        for (k, y) in enumerate(sample_waveforms)
            row = Any[sample_event_indices[k]]
            append!(row, y)
            println(io, join(row, ","))
        end
    end

    println("Wrote $(length(sample_waveforms)) sample waveforms to $path")
end

function plot_sample_waveforms(path, sample_waveforms, sample_event_indices)
    if isempty(sample_waveforms)
        println("No sample waveforms matched target contact $SAMPLE_TARGET_CONTACT and output contact $SAMPLE_OUTPUT_CONTACT")
        return
    end

    p = plot(
        title = "Sample waveforms: target contact $SAMPLE_TARGET_CONTACT, output contact $SAMPLE_OUTPUT_CONTACT",
        xlabel = "time / ns",
        ylabel = "baseline-subtracted signal / keV",
        size = (1100, 750),
        legend = :outerright,
        xlims = PLOT_XLIMS_NS
    )

    for (k, y) in enumerate(sample_waveforms)
        plot!(
            p,
            TIME_AXIS_NS,
            y,
            label = "event $(sample_event_indices[k])",
            linewidth = 1.3,
            alpha = 0.75
        )
    end

    savefig(p, path)
    println("Saved sample waveform plot to $path")
end

main()