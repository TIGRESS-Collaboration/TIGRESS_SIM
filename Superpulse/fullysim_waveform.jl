#
# Converts Geant4 hits into an ML-ready single-cluster waveform dataset:
#
#   geant_hits.csv
#       -> group hits by event
#       -> spatially cluster energy deposits
#       -> retain events with exactly one cluster
#       -> calculate energy-weighted cluster centroid and extent
#       -> simulate 9-contact SSD waveforms
#       -> apply fitted gain, alignment, shaping, delays, C and D matrices
#       -> store contiguous Float32 tensors and deterministic data splits
#
# Main ML arrays:
#   waveforms_keV       Float32 [N, 9, 100]
#   positions_mm        Float32 [N, 3]  columns: x, y, z_ssd
#   total_energy_keV    Float32 [N]
#   spatial_extent_mm   Float32 [N]
#   collecting_contact  Int8    [N]
#   event_id            Int64   [N]
#
# The waveforms are NOT normalized event by event. Their amplitudes remain in
# keV so energy information is preserved. Any later numerical scaling must use
# one fixed training-set scale, not a separate scale for each event.
# ============================================================

using Serialization
using Statistics
using LinearAlgebra
using Printf
using Random
using Unitful
using TypedTables
using SolidStateDetectors

# ============================ User settings ============================

const GEANT_HITS_FILE = "/data1/flerner/hpge_sims/geant/build/geant_hits.csv"

# Exact filename requested. If the actual detector builder wrote a different
# *_sim.jls filename, change only this path.
const SSD_SIM_FILE = "/data1/flerner/hpge_sims/CNN/gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls"

const FIT_FILE = "/data1/flerner/hpge_sims/CNN/joint_sim_to_real_fit/hierarchical_sim_to_real_fit.jls"

const OUT_DIR = "/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset"
const OUT_PREFIX = "gimpyB_single_cluster_waveforms"


const ALL_CONTACT_IDS = collect(1:9)
const OUTER_CONTACT_IDS = collect(1:8)
const CORE_CONTACT_ID = 9
const NCONTACTS = 9

# ============================ Clustering ============================

# Hits are connected when their Euclidean separation is at most this value.
# Clusters are connected components, so connectivity is transitive.
const HIT_CLUSTER_RADIUS_MM = 1.0

# Ignore numerically tiny energy deposits before clustering.
const MIN_HIT_ENERGY_KEV = 0.001

# Require exactly one connected spatial cluster.
const REQUIRE_EXACTLY_ONE_CLUSTER = true

# Optional additional compactness cut. Inf means no extra cut. The spatial
# extent is the energy-weighted RMS distance of hits from the cluster centroid.
const MAX_SPATIAL_EXTENT_MM = Inf

# ============================ Detector-edge exclusion ============================

# Reject clusters that lie too close to a detector boundary. Set to 0.0 to
# disable rejection while still calculating and storing edge distances.
const MIN_DETECTOR_EDGE_DISTANCE_MM = 1.0

# Require both the energy-weighted centroid and every retained Geant4 hit to
# satisfy the edge-distance cut. This is stricter and avoids accepting a
# centroid that is well inside while one cluster hit touches a boundary.
const REQUIRE_ALL_HITS_INSIDE_EDGE_MARGIN = true

# Gimpy B active-semiconductor geometry in SSD coordinates. These values match
# the detector builder used to create the serialized Simulation.
const DETECTOR_OUTER_RADIUS_MM = 30.0
const DETECTOR_OCTAGON_APOTHEM_MM = 29.0
const DETECTOR_Z_MIN_MM = 0.0
const DETECTOR_Z_MAX_MM = 90.0

# Offset rounded blind core bore.
const CORE_BORE_RADIUS_MM = 5.0
const CORE_BORE_CYLINDER_END_Z_MM = 70.0
const CORE_BORE_CAP_CENTER_Z_MM = 70.0
const CORE_BORE_CAP_TOP_Z_MM = 75.0
const CORE_BORE_OFFSET_X_MM = 29.0 - 27.2
const CORE_BORE_OFFSET_Y_MM = 0.0

# Taper starts at z = 60 mm. The exact taper consists of two planar cuts.
const DETECTOR_TAPER_START_Z_MM = 60.0
const DETECTOR_TAPER_ANGLE_DEG = 22.5

# Optional total-energy selection. This pipeline preserves all amplitudes.
const MIN_TOTAL_EDEP_KEV = 0.0
const MAX_TOTAL_EDEP_KEV = Inf

# ============================ SSD and output grid ============================

const SSD_SAMPLE_PERIOD_NS = 1.0
const SSD_MAX_NSTEPS = 2000
const SSD_DETECTOR_NUMBER = 1

const OUTPUT_SAMPLE_PERIOD_NS = 10.0
const OUTPUT_NSAMPLES = 100
const OUTPUT_TIME_NS = collect(0:OUTPUT_NSAMPLES-1) .* OUTPUT_SAMPLE_PERIOD_NS

const MAX_EVENTS = typemax(Int)
const PRINT_EVERY = 100

# ============================ ML split settings ============================

const TRAIN_FRACTION = 0.70
const VALIDATION_FRACTION = 0.15
const TEST_FRACTION = 0.15
const SPLIT_RANDOM_SEED = 74037

# Split approximately by collecting contact so all contacts are represented.
const STRATIFY_SPLITS_BY_CONTACT = true

# Gains used when the fit file does not contain simulation_amplitude_gains.
# Update these if the waveform fit is rerun.
const FALLBACK_TARGET_AMPLITUDE_GAINS = Dict(
    1 => 0.954641,
    2 => 0.956354,
    3 => 0.961302,
    4 => 0.964580,
    5 => 0.972914,
    6 => 0.982905,
    7 => 0.952198,
    8 => 0.947269
)

# ============================ Data types ============================

struct Hit
    track_id::Int
    x_mm::Float64
    y_mm::Float64
    z_g4_mm::Float64
    z_ssd_mm::Float64
    edep_keV::Float64
end

struct ClusterSummary
    hit_indices::Vector{Int}
    energy_keV::Float64
    centroid_mm::NTuple{3,Float64}
    sigma_xyz_mm::NTuple{3,Float64}
    spatial_extent_mm::Float64
    max_radius_mm::Float64
end

# ============================ Main ============================

function main()
    mkpath(OUT_DIR)
    validate_settings()

    isfile(GEANT_HITS_FILE) || error("Geant4 hit file not found: $GEANT_HITS_FILE")
    isfile(SSD_SIM_FILE) || error(
        "SSD simulation file not found: $SSD_SIM_FILE\n" *
        "Check whether the detector builder instead wrote " *
        "gimpy_B_constant_thickness_rounded_core_offset_bore_sim.jls."
    )
    isfile(FIT_FILE) || error("Fit file not found: $FIT_FILE")

    println("Loading SSD simulation...")
    sim = deserialize(SSD_SIM_FILE)

    println("Loading fitted waveform transformation...")
    fit = read_fit_parameters(deserialize(FIT_FILE))
    report_fit_parameters(fit)

    println("Reading Geant4 hits...")
    hit_events = read_geant_hits_csv(GEANT_HITS_FILE)
    all_event_ids = sort(collect(keys(hit_events)))
    if length(all_event_ids) > MAX_EVENTS
        resize!(all_event_ids, MAX_EVENTS)
    end

    waveforms = Matrix{Float32}[]
    positions = NTuple{3,Float32}[]
    sigma_xyz = NTuple{3,Float32}[]
    total_energies = Float32[]
    extents = Float32[]
    max_radii = Float32[]
    centroid_edge_distances = Float32[]
    minimum_hit_edge_distances = Float32[]
    collecting_contacts = Int8[]
    kept_event_ids = Int64[]
    raw_hit_counts = Int16[]

    rejection_counts = Dict(
        "zero_energy" => 0,
        "energy_window" => 0,
        "not_single_cluster" => 0,
        "extent_cut" => 0,
        "edge_distance_cut" => 0,
        "ssd_failure" => 0
    )
    failure_messages = Dict{Int,String}()

    nconsidered = 0

    for event_id in all_event_ids
        nconsidered += 1
        raw_hits = hit_events[event_id]
        hits = [h for h in raw_hits if h.edep_keV >= MIN_HIT_ENERGY_KEV]

        if isempty(hits)
            rejection_counts["zero_energy"] += 1
            continue
        end

        total_edep = sum(h.edep_keV for h in hits)
        if !(MIN_TOTAL_EDEP_KEV <= total_edep <= MAX_TOTAL_EDEP_KEV)
            rejection_counts["energy_window"] += 1
            continue
        end

        clusters = cluster_hits(hits, HIT_CLUSTER_RADIUS_MM)
        if REQUIRE_EXACTLY_ONE_CLUSTER && length(clusters) != 1
            rejection_counts["not_single_cluster"] += 1
            continue
        end

        # With REQUIRE_EXACTLY_ONE_CLUSTER=true exactly one entry is present.
        cluster = clusters[1]
        if cluster.spatial_extent_mm > MAX_SPATIAL_EXTENT_MM
            rejection_counts["extent_cut"] += 1
            continue
        end

        centroid_edge_distance = detector_edge_distance_mm(
            cluster.centroid_mm[1],
            cluster.centroid_mm[2],
            cluster.centroid_mm[3]
        )

        minimum_hit_edge_distance = minimum(
            detector_edge_distance_mm(h.x_mm, h.y_mm, h.z_ssd_mm)
            for h in hits
        )

        event_edge_distance = REQUIRE_ALL_HITS_INSIDE_EDGE_MARGIN ?
            min(centroid_edge_distance, minimum_hit_edge_distance) :
            centroid_edge_distance

        if !isfinite(event_edge_distance) ||
           event_edge_distance < MIN_DETECTOR_EDGE_DISTANCE_MM
            rejection_counts["edge_distance_cut"] += 1
            continue
        end

        try
            result = simulate_one_event(hits, event_id, sim)
            raw_native, native_time_ns = extract_contact_matrix(result)
            validate_contact_matrix(raw_native, native_time_ns, event_id)

            target = identify_target_contact(raw_native)
            processed = preprocess_for_fitted_model(
                raw_native, native_time_ns, target, fit
            )
            dS = derivative_matrix(processed, OUTPUT_SAMPLE_PERIOD_NS)
            corrected = fit.C_sets[target] * processed + fit.D_sets[target] * dS

            push!(waveforms, Float32.(corrected))
            push!(positions, Float32.(cluster.centroid_mm))
            push!(sigma_xyz, Float32.(cluster.sigma_xyz_mm))
            push!(total_energies, Float32(total_edep))
            push!(extents, Float32(cluster.spatial_extent_mm))
            push!(max_radii, Float32(cluster.max_radius_mm))
            push!(centroid_edge_distances, Float32(centroid_edge_distance))
            push!(minimum_hit_edge_distances, Float32(minimum_hit_edge_distance))
            push!(collecting_contacts, Int8(target))
            push!(kept_event_ids, Int64(event_id))
            push!(raw_hit_counts, Int16(length(hits)))
        catch err
            rejection_counts["ssd_failure"] += 1
            failure_messages[event_id] = sprint(showerror, err)
            @warn "SSD simulation or transformation failed" event_id=event_id exception=(err, catch_backtrace())
        end

        if nconsidered % PRINT_EVERY == 0
            println(
                "considered=$nconsidered kept=$(length(waveforms)) " *
                "rejections=$rejection_counts"
            )
        end
    end

    N = length(waveforms)
    N > 0 || error("No events survived clustering and SSD simulation")

    X = Array{Float32}(undef, N, NCONTACTS, OUTPUT_NSAMPLES)
    Y = Array{Float32}(undef, N, 3)
    Sigma = Array{Float32}(undef, N, 3)

    for n in 1:N
        X[n, :, :] .= waveforms[n]
        Y[n, :] .= collect(positions[n])
        Sigma[n, :] .= collect(sigma_xyz[n])
    end

    train_idx, validation_idx, test_idx = make_splits(
        collecting_contacts,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        SPLIT_RANDOM_SEED;
        stratify = STRATIFY_SPLITS_BY_CONTACT
    )

    # Fixed training-set statistics for optional numerical standardization.
    # These do not alter waveforms_keV. The ML loader may use them consistently
    # for train, validation, test, and deployment data.
    train_channel_mean, train_channel_std = channel_statistics(X, train_idx)

    dataset = Dict(
        "waveforms_keV" => X,
        "positions_mm" => Y,
        "position_columns" => ["x_mm", "y_mm", "z_ssd_mm"],
        "hit_sigma_xyz_mm" => Sigma,
        "total_energy_keV" => total_energies,
        "spatial_extent_mm" => extents,
        "max_cluster_radius_mm" => max_radii,
        "centroid_edge_distance_mm" => centroid_edge_distances,
        "minimum_hit_edge_distance_mm" => minimum_hit_edge_distances,
        "collecting_contact" => collecting_contacts,
        "event_id" => kept_event_ids,
        "raw_hit_count" => raw_hit_counts,
        "time_ns" => Float32.(OUTPUT_TIME_NS),
        "contact_ids" => Int8.(ALL_CONTACT_IDS),
        "train_indices" => train_idx,
        "validation_indices" => validation_idx,
        "test_indices" => test_idx,
        "train_channel_mean_keV" => train_channel_mean,
        "train_channel_std_keV" => train_channel_std,
        "rejection_counts" => rejection_counts,
        "failure_messages" => failure_messages,
        "metadata" => Dict(
            "dataset_version" => 1,
            "geant_hits_file" => GEANT_HITS_FILE,
            "ssd_sim_file" => SSD_SIM_FILE,
            "fit_file" => FIT_FILE,
            "waveform_shape" => (NCONTACTS, OUTPUT_NSAMPLES),
            "waveform_dtype" => "Float32",
            "waveform_unit" => "keV",
            "eventwise_amplitude_normalization" => false,
            "label_definition" => "energy-weighted centroid of the single spatial hit cluster",
            "coordinate_system" => "SSD Cartesian coordinates",
            "cluster_algorithm" => "connected components under Euclidean distance threshold",
            "cluster_radius_mm" => HIT_CLUSTER_RADIUS_MM,
            "minimum_hit_energy_keV" => MIN_HIT_ENERGY_KEV,
            "maximum_spatial_extent_mm" => MAX_SPATIAL_EXTENT_MM,
            "minimum_detector_edge_distance_mm" => MIN_DETECTOR_EDGE_DISTANCE_MM,
            "require_all_hits_inside_edge_margin" => REQUIRE_ALL_HITS_INSIDE_EDGE_MARGIN,
            "edge_distance_boundaries" => [
                "outer cylinder",
                "octagonal flats",
                "top and bottom faces",
                "two top taper planes",
                "offset rounded core bore"
            ],
            "single_cluster_required" => REQUIRE_EXACTLY_ONE_CLUSTER,
            "sample_period_ns" => OUTPUT_SAMPLE_PERIOD_NS,
            "split_seed" => SPLIT_RANDOM_SEED,
            "split_fractions" => (
                TRAIN_FRACTION,
                VALIDATION_FRACTION,
                TEST_FRACTION
            ),
            "splits_stratified_by_contact" => STRATIFY_SPLITS_BY_CONTACT,
            "recommended_model_input_order" => "event, contact, time",
            "recommended_initial_output" => "x_mm, y_mm, z_ssd_mm"
        )
    )

    output_path = joinpath(OUT_DIR, "$(OUT_PREFIX).jls")
    serialize(output_path, dataset)

    write_manifest_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_manifest.csv"),
        kept_event_ids,
        collecting_contacts,
        total_energies,
        positions,
        sigma_xyz,
        extents,
        max_radii,
        centroid_edge_distances,
        minimum_hit_edge_distances,
        raw_hit_counts,
        train_idx,
        validation_idx,
        test_idx
    )

    write_readme(joinpath(OUT_DIR, "$(OUT_PREFIX)_README.txt"), N)

    println("\nDataset complete.")
    println("  retained events: $N")
    println("  train:      $(length(train_idx))")
    println("  validation: $(length(validation_idx))")
    println("  test:       $(length(test_idx))")
    println("  rejection counts: $rejection_counts")
    println("  saved: $output_path")
    println("  tensor shape: $(size(X)) = event x contact x time")
end

# ============================ Spatial clustering ============================

# Signed distance margins to the analytic detector boundaries. Positive values
# are inside the active semiconductor. Negative values are outside or inside
# the excluded core-bore volume. The minimum positive margin is the distance to
# the nearest modeled boundary.
function detector_edge_distance_mm(x::Real, y::Real, z::Real)
    xf = Float64(x)
    yf = Float64(y)
    zf = Float64(z)

    radial = hypot(xf, yf)
    circle_margin = DETECTOR_OUTER_RADIUS_MM - radial

    # The outer shape is the intersection of an axis-aligned square and a
    # square rotated by 45 degrees, in addition to the radius-30 cylinder.
    square_margin = DETECTOR_OCTAGON_APOTHEM_MM - max(abs(xf), abs(yf))
    rotated_x = (xf + yf) / sqrt(2.0)
    rotated_y = (-xf + yf) / sqrt(2.0)
    rotated_square_margin = DETECTOR_OCTAGON_APOTHEM_MM -
                            max(abs(rotated_x), abs(rotated_y))

    bottom_margin = zf - DETECTOR_Z_MIN_MM
    top_margin = DETECTOR_Z_MAX_MM - zf

    # The two top taper cuts move the negative-x and positive-y boundaries
    # inward above z = 60 mm. At z=60 they meet the corresponding octagon flats.
    taper_dx = max(zf - DETECTOR_TAPER_START_Z_MM, 0.0) *
               tand(DETECTOR_TAPER_ANGLE_DEG)
    negative_x_taper_margin = xf + DETECTOR_OCTAGON_APOTHEM_MM - taper_dx
    positive_y_taper_margin = DETECTOR_OCTAGON_APOTHEM_MM - taper_dx - yf

    bore_margin = distance_from_core_bore_mm(xf, yf, zf)

    return minimum((
        circle_margin,
        square_margin,
        rotated_square_margin,
        bottom_margin,
        top_margin,
        negative_x_taper_margin,
        positive_y_taper_margin,
        bore_margin
    ))
end

function distance_from_core_bore_mm(x::Real, y::Real, z::Real)
    dx = Float64(x) - CORE_BORE_OFFSET_X_MM
    dy = Float64(y) - CORE_BORE_OFFSET_Y_MM
    r = hypot(dx, dy)
    zf = Float64(z)

    if zf <= CORE_BORE_CYLINDER_END_Z_MM
        # Outside the cylindrical bore, positive distance is r - radius.
        return r - CORE_BORE_RADIUS_MM
    else
        # Above the cylinder, the rounded blind end is a radius-5 sphere
        # centered at z=70 mm. This also provides the nearest-bore distance for
        # points above the cap.
        center_distance = hypot(r, zf - CORE_BORE_CAP_CENTER_Z_MM)
        return center_distance - CORE_BORE_RADIUS_MM
    end
end

squared_distance_mm(a::Hit, b::Hit) =
    (a.x_mm - b.x_mm)^2 +
    (a.y_mm - b.y_mm)^2 +
    (a.z_ssd_mm - b.z_ssd_mm)^2

function cluster_hits(hits, radius_mm)
    n = length(hits)
    visited = falses(n)
    radius2 = radius_mm^2
    components = Vector{Vector{Int}}()

    for seed in 1:n
        visited[seed] && continue
        component = Int[]
        queue = [seed]
        visited[seed] = true

        while !isempty(queue)
            i = popfirst!(queue)
            push!(component, i)

            for j in 1:n
                if !visited[j] && squared_distance_mm(hits[i], hits[j]) <= radius2
                    visited[j] = true
                    push!(queue, j)
                end
            end
        end

        push!(components, component)
    end

    summaries = [summarize_cluster(hits, indices) for indices in components]
    sort!(summaries, by = c -> c.energy_keV, rev = true)
    return summaries
end

function summarize_cluster(hits, indices)
    energies = [hits[i].edep_keV for i in indices]
    total = sum(energies)
    total > 0 || error("Cluster has non-positive deposited energy")

    x = sum(energies[k] * hits[indices[k]].x_mm for k in eachindex(indices)) / total
    y = sum(energies[k] * hits[indices[k]].y_mm for k in eachindex(indices)) / total
    z = sum(energies[k] * hits[indices[k]].z_ssd_mm for k in eachindex(indices)) / total

    var_x = sum(energies[k] * (hits[indices[k]].x_mm - x)^2 for k in eachindex(indices)) / total
    var_y = sum(energies[k] * (hits[indices[k]].y_mm - y)^2 for k in eachindex(indices)) / total
    var_z = sum(energies[k] * (hits[indices[k]].z_ssd_mm - z)^2 for k in eachindex(indices)) / total

    extent = sqrt(max(var_x + var_y + var_z, 0.0))
    max_radius = maximum(
        sqrt(
            (hits[i].x_mm - x)^2 +
            (hits[i].y_mm - y)^2 +
            (hits[i].z_ssd_mm - z)^2
        )
        for i in indices
    )

    return ClusterSummary(
        copy(indices),
        total,
        (x, y, z),
        (sqrt(max(var_x, 0.0)), sqrt(max(var_y, 0.0)), sqrt(max(var_z, 0.0))),
        extent,
        max_radius
    )
end

# ============================ Geant4 CSV ============================

function read_geant_hits_csv(path)
    events = Dict{Int,Vector{Hit}}()

    open(path, "r") do io
        header = strip.(split(strip(readline(io)), ','))
        required = [
            "event_id", "track_id", "x_mm", "y_mm",
            "z_g4_mm", "z_ssd_mm", "edep_keV"
        ]
        index = Dict(name => findfirst(==(name), header) for name in required)
        any(isnothing, values(index)) &&
            error("CSV must contain columns $(join(required, ", ")). Found $header")

        for (line_offset, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            fields = strip.(split(strip(line), ','))
            try
                event_id = parse(Int, fields[index["event_id"]])
                hit = Hit(
                    parse(Int, fields[index["track_id"]]),
                    parse(Float64, fields[index["x_mm"]]),
                    parse(Float64, fields[index["y_mm"]]),
                    parse(Float64, fields[index["z_g4_mm"]]),
                    parse(Float64, fields[index["z_ssd_mm"]]),
                    parse(Float64, fields[index["edep_keV"]])
                )
                push!(get!(events, event_id, Hit[]), hit)
            catch err
                error("Failed to parse line $line_number: $line\n$(sprint(showerror, err))")
            end
        end
    end

    return events
end

# ============================ SSD event simulation ============================

function hits_to_ssd_event(
    hits,
    event_id;
    T::Type{<:AbstractFloat} = Float32
)
    positions = Vector{CartesianPoint{T}}(
        undef,
        length(hits)
    )

    energies = Vector{
        typeof(T(1) * u"keV")
    }(
        undef,
        length(hits)
    )

    times = Vector{
        typeof(T(1) * u"ns")
    }(
        undef,
        length(hits)
    )

    detector_numbers = fill(
        SSD_DETECTOR_NUMBER,
        length(hits)
    )

    for i in eachindex(hits)
        hit = hits[i]

        positions[i] = CartesianPoint{T}(
            T(hit.x_mm * 1e-3),
            T(hit.y_mm * 1e-3),
            T(hit.z_ssd_mm * 1e-3)
        )

        energies[i] = T(hit.edep_keV) * u"keV"
        times[i] = T(0) * u"ns"
    end

    event_table = Table((
        evtno = Int[event_id],
        detno = [detector_numbers],
        thit = [times],
        edep = [energies],
        pos = [positions]
    ))

    return event_table
end

function simulate_one_event(
    hits,
    event_id,
    sim
)
    T = Float32

    event_table = hits_to_ssd_event(
        hits,
        event_id;
        T = T
    )

    return simulate_waveforms(
        event_table,
        sim;
        Δt = T(SSD_SAMPLE_PERIOD_NS) * u"ns",
        signal_unit = u"keV",
        max_nsteps = SSD_MAX_NSTEPS,
        max_interaction_distance = T(0.001),
        verbose = false
    )
end

# ============================ SSD waveform extraction ============================

function numeric_vector(x)
    if x isa AbstractVector{<:Number}
        return Float64.(ustrip.(x))
    elseif hasproperty(x, :signal)
        return Float64.(ustrip.(collect(getproperty(x, :signal))))
    elseif hasproperty(x, :values)
        return Float64.(ustrip.(collect(getproperty(x, :values))))
    elseif hasproperty(x, :samples)
        return Float64.(ustrip.(collect(getproperty(x, :samples))))
    else
        try
            return Float64.(ustrip.(collect(x)))
        catch
            error("Unsupported waveform object $(typeof(x)); properties=$(propertynames(x))")
        end
    end
end

function waveform_payload(result)
    hasproperty(result, :waveform) && return getproperty(result, :waveform)
    result isa AbstractDict && haskey(result, "waveform") && return result["waveform"]
    result isa AbstractDict && haskey(result, :waveform) && return result[:waveform]
    error("SSD result has no waveform field. Type=$(typeof(result))")
end

function extract_contact_matrix(result)
    payload = waveform_payload(result)
    if payload isa AbstractVector && length(payload) == 1
        payload = payload[1]
    end

    matrix = nothing

    if payload isa AbstractMatrix
        a = Float64.(ustrip.(payload))
        matrix = size(a, 1) == NCONTACTS ? a :
                 size(a, 2) == NCONTACTS ? permutedims(a) : nothing
    elseif payload isa AbstractVector && length(payload) == NCONTACTS
        channels = numeric_vector.(payload)
        n = length(channels[1])
        all(length(ch) == n for ch in channels) ||
            error("SSD contact waveforms have unequal lengths")
        matrix = reduce(vcat, permutedims.(channels))
    elseif hasproperty(payload, :waveforms)
        channels = collect(getproperty(payload, :waveforms))
        length(channels) == NCONTACTS ||
            error("Expected 9 contacts, found $(length(channels))")
        matrix = reduce(vcat, permutedims.(numeric_vector.(channels)))
    end

    matrix === nothing && error(
        "Could not extract a 9-contact SSD waveform matrix. " *
        "Payload type=$(typeof(payload)), properties=$(propertynames(payload))"
    )

    native_t = collect(0:size(matrix, 2)-1) .* SSD_SAMPLE_PERIOD_NS
    return Matrix{Float64}(matrix), native_t
end

function validate_contact_matrix(S, t, event_id)
    size(S, 1) == NCONTACTS ||
        error("Event $event_id produced $(size(S,1)) contacts rather than 9")
    size(S, 2) == length(t) || error("Event $event_id waveform/time mismatch")
    all(isfinite, S) || error("Event $event_id contains nonfinite waveform values")
end

# ============================ Fitted transformation ============================

function get_required(d, key)
    haskey(d, key) && return d[key]
    haskey(d, Symbol(key)) && return d[Symbol(key)]
    error("Missing key '$key'. Available: $(collect(keys(d)))")
end

function get_optional(d, key, default)
    haskey(d, key) && return d[key]
    haskey(d, Symbol(key)) && return d[Symbol(key)]
    return default
end

function normalize_target_matrix_dict(x)
    out = Dict{Int,Matrix{Float64}}()
    for (k, v) in pairs(x)
        target = k isa Integer ? Int(k) : parse(Int, String(k))
        out[target] = Matrix{Float64}(v)
    end
    return out
end

function normalize_float_dict(x)
    out = Dict{Int,Float64}()
    for (k, v) in pairs(x)
        key = k isa Integer ? Int(k) : parse(Int, String(k))
        out[key] = Float64(v)
    end
    return out
end

function read_fit_parameters(data)
    C_sets = normalize_target_matrix_dict(get_required(data, "target_integral_crosstalk"))
    D_sets = normalize_target_matrix_dict(get_required(data, "target_differential_crosstalk_ns"))
    global_shifts = normalize_float_dict(get_required(data, "global_shifts_ns"))
    channel_delays = Float64.(get_required(data, "channel_delays_ns"))
    metadata = get_optional(data, "metadata", Dict{String,Any}())
    flip_core = Bool(get_optional(metadata, "flip_sim_core_polarity", true))
    gains = normalize_float_dict(get_optional(
        data, "simulation_amplitude_gains", FALLBACK_TARGET_AMPLITUDE_GAINS
    ))

    for target in OUTER_CONTACT_IDS
        haskey(C_sets, target) || error("Missing target C matrix $target")
        haskey(D_sets, target) || error("Missing target D matrix $target")
        haskey(global_shifts, target) || error("Missing target shift $target")
        haskey(gains, target) || error("Missing target gain $target")
        size(C_sets[target]) == (9, 9) || error("C target $target is not 9x9")
        size(D_sets[target]) == (9, 9) || error("D target $target is not 9x9")
    end
    length(channel_delays) == 9 || error("Expected 9 channel delays")

    return (
        C_sets = C_sets,
        D_sets = D_sets,
        global_shifts = global_shifts,
        channel_delays = channel_delays,
        amplitude_gains = gains,
        outer_tau = Float64(get_required(data, "outer_tau_ns")),
        core_tau = Float64(get_required(data, "core_tau_ns")),
        core_delay = Float64(get_required(data, "core_delay_ns")),
        flip_core = flip_core
    )
end

function report_fit_parameters(fit)
    @printf("  outer tau: %.3f ns\n", fit.outer_tau)
    @printf("  core tau: %.3f ns\n", fit.core_tau)
    @printf("  core delay: %.3f ns\n", fit.core_delay)
    println("  channel delays: $(fit.channel_delays)")
    println("  amplitude gains: $(fit.amplitude_gains)")
end

function cid_to_index(cid)
    i = findfirst(==(cid), ALL_CONTACT_IDS)
    isnothing(i) && error("Unknown contact $cid")
    return i
end

function identify_target_contact(S)
    n = size(S, 2)
    lo = max(1, floor(Int, 0.90 * n))
    levels = [mean(@view S[cid_to_index(cid), lo:n]) for cid in OUTER_CONTACT_IDS]
    return OUTER_CONTACT_IDS[argmax(abs.(levels))]
end

function interp_constant_edges(x, y, xnew)
    out = Vector{Float64}(undef, length(xnew))
    j = 1
    for i in eachindex(xnew)
        xx = xnew[i]
        if xx <= x[1]
            out[i] = y[1]
        elseif xx >= x[end]
            out[i] = y[end]
        else
            while j < length(x)-1 && x[j+1] < xx
                j += 1
            end
            f = (xx - x[j]) / (x[j+1] - x[j])
            out[i] = y[j] + f * (y[j+1] - y[j])
        end
    end
    return out
end

sample_shifted(source_t, y, target_t, shift_ns) =
    interp_constant_edges(source_t, y, target_t .- shift_ns)

function lowpass_onepole(y, dt_ns, tau_ns)
    tau_ns <= 0 && return copy(y)
    a = exp(-dt_ns / tau_ns)
    out = similar(y)
    out[1] = y[1]
    for k in 2:length(y)
        out[k] = a * out[k-1] + (1-a) * y[k]
    end
    return out
end

function lowpass_twopole(y, dt_ns, tau_ns)
    tau_ns <= 0 && return copy(y)
    return lowpass_onepole(lowpass_onepole(y, dt_ns, tau_ns), dt_ns, tau_ns)
end

function preprocess_for_fitted_model(S_native, native_t, target, fit)
    S = copy(S_native)
    if fit.flip_core
        S[cid_to_index(CORE_CONTACT_ID), :] .*= -1
    end

    # This is a fixed per-contact-class sim-to-real calibration. It does not
    # normalize individual event amplitudes, so energy information is retained.
    S .*= fit.amplitude_gains[target]

    out = zeros(Float64, NCONTACTS, OUTPUT_NSAMPLES)
    for cid in ALL_CONTACT_IDS
        i = cid_to_index(cid)
        tau = cid == CORE_CONTACT_ID ? fit.core_tau : fit.outer_tau
        local_delay = fit.channel_delays[i]
        cid == CORE_CONTACT_ID && (local_delay += fit.core_delay)
        total_shift = fit.global_shifts[target] + local_delay

        y = sample_shifted(native_t, vec(S[i, :]), OUTPUT_TIME_NS, total_shift)
        out[i, :] .= lowpass_twopole(y, OUTPUT_SAMPLE_PERIOD_NS, tau)
    end
    return out
end

function derivative_matrix(S, dt_ns)
    d = zeros(size(S))
    n = size(S, 2)
    for ch in axes(S, 1)
        if n == 1
            d[ch, 1] = 0.0
        elseif n == 2
            d[ch, 1] = (S[ch, 2] - S[ch, 1]) / dt_ns
            d[ch, 2] = d[ch, 1]
        else
            d[ch, 1] = (S[ch, 2] - S[ch, 1]) / dt_ns
            for k in 2:n-1
                d[ch, k] = (S[ch, k+1] - S[ch, k-1]) / (2 * dt_ns)
            end
            d[ch, n] = (S[ch, n] - S[ch, n-1]) / dt_ns
        end
    end
    return d
end

# ============================ ML splits/statistics ============================

function make_splits(contacts, train_fraction, validation_fraction, seed; stratify=true)
    rng = MersenneTwister(seed)
    train = Int[]
    validation = Int[]
    test = Int[]

    groups = if stratify
        [findall(==(Int8(cid)), contacts) for cid in OUTER_CONTACT_IDS]
    else
        [collect(eachindex(contacts))]
    end

    for group in groups
        isempty(group) && continue
        shuffled = shuffle(rng, group)
        n = length(shuffled)
        ntrain = floor(Int, train_fraction * n)
        nvalidation = floor(Int, validation_fraction * n)

        append!(train, shuffled[1:ntrain])
        append!(validation, shuffled[ntrain+1:ntrain+nvalidation])
        append!(test, shuffled[ntrain+nvalidation+1:end])
    end

    shuffle!(rng, train)
    shuffle!(rng, validation)
    shuffle!(rng, test)
    return train, validation, test
end

function channel_statistics(X, train_indices)
    isempty(train_indices) && error("Training split is empty")
    means = zeros(Float32, NCONTACTS)
    stds = zeros(Float32, NCONTACTS)

    for channel in 1:NCONTACTS
        values = vec(X[train_indices, channel, :])
        means[channel] = Float32(mean(values))
        stds[channel] = Float32(max(std(values), 1f-6))
    end
    return means, stds
end

# ============================ Output ============================

function split_name(index, train_set, validation_set, test_set)
    index in train_set && return "train"
    index in validation_set && return "validation"
    index in test_set && return "test"
    return "unknown"
end

function write_manifest_csv(path, event_ids, contacts, energies, positions,
                            sigma_xyz, extents, max_radii,
                            centroid_edge_distances, minimum_hit_edge_distances,
                            hit_counts, train_idx, validation_idx, test_idx)
    train_set = Set(train_idx)
    validation_set = Set(validation_idx)
    test_set = Set(test_idx)

    open(path, "w") do io
        println(io,
            "dataset_index,event_id,split,collecting_contact,total_energy_keV," *
            "x_mm,y_mm,z_ssd_mm,sigma_x_mm,sigma_y_mm,sigma_z_mm," *
            "spatial_extent_mm,max_cluster_radius_mm," *
            "centroid_edge_distance_mm,minimum_hit_edge_distance_mm,raw_hit_count")

        for i in eachindex(event_ids)
            p = positions[i]
            s = sigma_xyz[i]
            split = split_name(i, train_set, validation_set, test_set)
            println(io, join((
                i, event_ids[i], split, contacts[i], energies[i],
                p[1], p[2], p[3], s[1], s[2], s[3],
                extents[i], max_radii[i], centroid_edge_distances[i],
                minimum_hit_edge_distances[i], hit_counts[i]
            ), ","))
        end
    end
end

function write_readme(path, N)
    open(path, "w") do io
        println(io, "ML dataset: $OUT_PREFIX")
        println(io, "Events: $N")
        println(io, "")
        println(io, "Load in Julia:")
        println(io, "  using Serialization")
        println(io, "  d = deserialize(\"$(OUT_PREFIX).jls\")")
        println(io, "  X = d[\"waveforms_keV\"]      # N x 9 x 100 Float32")
        println(io, "  y = d[\"positions_mm\"]       # N x 3 Float32")
        println(io, "  train = d[\"train_indices\"]")
        println(io, "")
        println(io, "Waveforms retain amplitudes in keV. Do not divide each event")
        println(io, "by its own maximum. Optional fixed standardization values are")
        println(io, "stored in train_channel_mean_keV and train_channel_std_keV.")
        println(io, "")
        println(io, "The initial model output should be x_mm, y_mm, z_ssd_mm.")
        println(io, "Every retained event contains exactly one spatial hit cluster.")
        println(io, "Minimum detector-edge distance: $(MIN_DETECTOR_EDGE_DISTANCE_MM) mm")
        println(io, "Edge distances are stored for the centroid and nearest hit.")
    end
end

function validate_settings()
    abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) < 1e-9 ||
        error("Split fractions must sum to one")
    HIT_CLUSTER_RADIUS_MM > 0 || error("HIT_CLUSTER_RADIUS_MM must be positive")
    MIN_DETECTOR_EDGE_DISTANCE_MM >= 0 ||
        error("MIN_DETECTOR_EDGE_DISTANCE_MM must be nonnegative")
    DETECTOR_Z_MIN_MM < DETECTOR_Z_MAX_MM || error("Invalid detector z limits")
    OUTPUT_NSAMPLES > 1 || error("OUTPUT_NSAMPLES must exceed one")
end

main()