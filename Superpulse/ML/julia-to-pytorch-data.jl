#
# Converts the Julia-serialized ML dataset into C-order binary arrays for
# NumPy/PyTorch. Julia is column-major, so multidimensional arrays are
# explicitly permuted before writing.
# ============================================================

using Serialization
using Printf

const INPUT_JLS = "/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset/gimpyB_single_cluster_waveforms.jls"
const OUTPUT_DIR = "/data1/flerner/hpge_sims/CNN/ml_single_cluster_dataset/pytorch_export_v2"

function write_vector(path, x)
    open(path, "w") do io
        write(io, vec(x))
    end
end

function write_matrix_c_order(path, x)
    ndims(x) == 2 || error("Expected a matrix for $path")
    # Julia writes the first dimension fastest. Writing D x N makes NumPy's
    # C-order reshape to N x D recover the original matrix.
    write_vector(path, permutedims(x, (2, 1)))
end

function write_tensor3_c_order(path, x)
    ndims(x) == 3 || error("Expected a rank-3 tensor for $path")
    # Original layout: N x C x T. Write T x C x N so NumPy C-order reads
    # sample fastest, then contact, then event.
    write_vector(path, permutedims(x, (3, 2, 1)))
end

function write_indices(path, indices)
    # Julia is 1-based; Python is 0-based.
    write_vector(path, Int64.(indices .- 1))
end

function main()
    isfile(INPUT_JLS) || error("Dataset not found: $INPUT_JLS")
    mkpath(OUTPUT_DIR)

    println("Loading Julia dataset:")
    println("  $INPUT_JLS")
    d = deserialize(INPUT_JLS)

    X = Array{Float32,3}(d["waveforms_keV"])
    y = Array{Float32,2}(d["positions_mm"])
    energy = Float32.(d["total_energy_keV"])
    contact = Int8.(d["collecting_contact"])
    event_id = Int64.(d["event_id"])
    extent = Float32.(d["spatial_extent_mm"])
    time_ns = Float32.(d["time_ns"])
    channel_mean = Float32.(d["train_channel_mean_keV"])
    channel_std = Float32.(d["train_channel_std_keV"])

    N, C, T = size(X)
    C == 9 || error("Expected 9 contacts, found $C")
    size(y) == (N, 3) || error("positions_mm must have shape N x 3")
    length(energy) == N || error("Energy length mismatch")
    length(contact) == N || error("Contact length mismatch")
    length(event_id) == N || error("Event ID length mismatch")

    write_tensor3_c_order(joinpath(OUTPUT_DIR, "waveforms_f32.bin"), X)
    write_matrix_c_order(joinpath(OUTPUT_DIR, "positions_f32.bin"), y)
    write_vector(joinpath(OUTPUT_DIR, "energy_f32.bin"), energy)
    write_vector(joinpath(OUTPUT_DIR, "contact_i8.bin"), contact)
    write_vector(joinpath(OUTPUT_DIR, "event_id_i64.bin"), event_id)
    write_vector(joinpath(OUTPUT_DIR, "extent_f32.bin"), extent)
    write_vector(joinpath(OUTPUT_DIR, "time_f32.bin"), time_ns)
    write_vector(joinpath(OUTPUT_DIR, "channel_mean_f32.bin"), channel_mean)
    write_vector(joinpath(OUTPUT_DIR, "channel_std_f32.bin"), channel_std)
    write_indices(joinpath(OUTPUT_DIR, "train_indices_i64.bin"), d["train_indices"])
    write_indices(joinpath(OUTPUT_DIR, "validation_indices_i64.bin"), d["validation_indices"])
    write_indices(joinpath(OUTPUT_DIR, "test_indices_i64.bin"), d["test_indices"])

    open(joinpath(OUTPUT_DIR, "metadata.txt"), "w") do io
        println(io, "n_events=$N")
        println(io, "n_contacts=$C")
        println(io, "n_samples=$T")
        println(io, "position_dims=3")
        println(io, "waveform_unit=keV")
        println(io, "position_unit=mm")
        println(io, "array_order=C")
        println(io, "tensor_order=event,contact,time")
        println(io, "export_version=2")
    end

    # Human-readable values for the Python round-trip check.
    open(joinpath(OUTPUT_DIR, "roundtrip_reference.txt"), "w") do io
        println(io, "X_shape=$(size(X))")
        println(io, "y_shape=$(size(y))")
        println(io, "X_event1_contact1_first10=$(join(X[1,1,1:min(10,T)], ','))")
        println(io, "X_event1_contact9_first10=$(join(X[1,9,1:min(10,T)], ','))")
        println(io, "y_event1=$(join(y[1,:], ','))")
        println(io, "event_id_1=$(event_id[1])")
        println(io, "contact_1=$(contact[1])")
        println(io, "energy_1=$(energy[1])")
    end

    println("\nExport complete:")
    println("  events: $N")
    println("  waveform tensor: ($N, $C, $T)")
    println("  output: $OUTPUT_DIR")
    for contact in (1, 9)
        ytrace = X[1, contact, :]
        peak_index = argmax(abs.(ytrace))

        lo = max(1, peak_index - 5)
        hi = min(T, peak_index + 5)

        println(
            "event 1, contact $contact, peak index = $peak_index, " *
            "peak value = $(ytrace[peak_index])"
        )

        println(
            "X[1,$contact,$lo:$hi] = ",
            ytrace[lo:hi]
        )
    end

    println("y[1,:] = ", y[1, :])
end

main()