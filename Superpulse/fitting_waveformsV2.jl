# fitting_waveformsV2.jl
# Joint fit of simulated HPGe superpulses to real data with:
#   - native-grid baseline subtraction
#   - explicit simulated-core polarity correction
#   - derivative cross-correlation target alignment
#   - shared outer-contact shaping
#   - shared core shaping and core delay
#   - per-channel residual delay corrections
#   - hierarchical integral and differential crosstalk
#   - shared electronics matrices plus target-specific corrections
#   - collecting/core identity priors only, not mirror-channel identity priors
#   - robust ridge regression
#   - diagnostics, plots, CSVs, and serialized output
# ============================================================

using Serialization
using Statistics
using LinearAlgebra
using Printf

ENV["GKSwstype"] = "100"
using Plots
gr()
closeall()
default(background_color=:white, background_color_inside=:white,
        foreground_color=:black, dpi=150)

# ============================ User settings ============================

const REAL_FILE = "/data1/flerner/hpge_sims/CNN/real_waveform_only_superpulses/real_waveform_only_superpulses.jls"
const SIM_FILE = "/data1/flerner/hpge_sims/CNN/gimpytests/repulsion_gimpy_geant_ssd_superpulses.jls"
const OUT_DIR = "/data1/flerner/hpge_sims/CNN/joint_sim_to_real_fit"
const OUT_PREFIX = "hierarchical_sim_to_real_fit"

const TARGET_CONTACTS = collect(1:8)
const ALL_CONTACT_IDS = collect(1:9)
const CORE_CONTACT_ID = 9
const NCONTACTS = 9

const REAL_SAMPLE_PERIOD_NS = 10.0
const REAL_NSAMPLES = 100
const FLIP_SIM_CORE_POLARITY = true

const REAL_BASELINE_END_NS = 290.0
const SIM_BASELINE_END_NS = 50.0

const ALIGN_REFERENCE_CONTACT = CORE_CONTACT_ID
const ALIGN_FRACTION = 0.10
const XC_ORIGINAL_MIN_SHIFT_NS = -400.0
const XC_ORIGINAL_MAX_SHIFT_NS = 400.0
const XC_ORIGINAL_STEP_NS = 1.0
const REFINE_SHIFT_HALF_WINDOW_NS = 50.0
const REFINE_SHIFT_STEP_NS = 0.5

const FIT_START_NS = 250.0
const FIT_END_NS = 950.0
const EDGE_START_NS = 300.0
const EDGE_END_NS = 500.0

const OUTER_TAU_MIN_NS = 0.0
const OUTER_TAU_MAX_NS = 200.0
const CORE_TAU_MIN_NS = 0.0
const CORE_TAU_MAX_NS = 200.0
const CORE_DELAY_MIN_NS = -100.0
const CORE_DELAY_MAX_NS = 100.0
const COARSE_STEP_NS = 10.0
const FINE_STEP_NS = 1.0

# Residual per-channel delay search. These delays are fitted after the shared
# shaping and shared core delay have been determined.
const FIT_CHANNEL_DELAYS = true
const CHANNEL_DELAY_MIN_NS = -20.0
const CHANNEL_DELAY_MAX_NS = 20.0
const CHANNEL_DELAY_COARSE_STEP_NS = 2.0
const CHANNEL_DELAY_FINE_STEP_NS = 0.25

# The shared core delay already represents the core-versus-contact electronics
# delay. Keep the residual core-channel delay fixed at zero to avoid fitting two
# mathematically redundant core delays.
const FIX_CORE_RESIDUAL_DELAY = true

const NORMALIZE_SIM_AMPLITUDES = false

const RIDGE_DIAGONAL = 1.0e-3
const RIDGE_CONTACT_TO_CONTACT = 3.0
const RIDGE_CORE_TO_CONTACT = 1.0
const RIDGE_CONTACT_TO_CORE = 1.0
# Derivative columns are multiplied by this timescale before regression so
# their numerical scale is comparable to the waveform columns. D is converted
# back to physical nanoseconds after fitting.
const DERIVATIVE_REFERENCE_NS = 10.0
const RIDGE_DIFFERENTIAL_DIAGONAL = 0.1
const RIDGE_DIFFERENTIAL_OFFDIAGONAL = 1.0

# Hierarchical target-specific correction penalties. Larger values force the
# target matrices closer to the shared electronics matrices. The integral
# correction has dimensionless coefficients. The differential correction is
# fitted on derivative columns scaled by DERIVATIVE_REFERENCE_NS.
const FIT_TARGET_SPECIFIC_CORRECTIONS = true
const TARGET_RIDGE_INTEGRAL_COLLECTING = 0.25
const TARGET_RIDGE_INTEGRAL_CORE = 0.50
const TARGET_RIDGE_INTEGRAL_MIRROR = 0.75
const TARGET_RIDGE_DIFFERENTIAL = 1.50
const TARGET_ROBUST_IRLS_ITERATIONS = 6

# Optional coefficient warnings. These do not clip the fit.
const WARN_TARGET_INTEGRAL_CORRECTION_ABS = 0.50
const WARN_TARGET_DIFFERENTIAL_CORRECTION_NS_ABS = 10.0

const ROBUST_CROSSTALK_FIT = true
const ROBUST_IRLS_ITERATIONS = 5
const HUBER_K = 1.345

const WARN_INTEGRAL_CROSSTALK_ABS = 0.10
const WARN_DIFFERENTIAL_CROSSTALK_NS_ABS = 20.0
const WARN_CHANNEL_DELAY_NS_ABS = 15.0

const NORMALIZE_RECEIVER_CHANNELS = true
const MIN_CHANNEL_SCALE_KEV = 10.0

# ============================ Main ============================

function main()
    mkpath(OUT_DIR)
    real_data = deserialize(REAL_FILE)
    sim_data = deserialize(SIM_FILE)
    real_superpulses = get_required(real_data, "superpulses")
    sim_superpulses = get_required(sim_data, "superpulses")
    real_time = read_time_axis(real_data, :real)
    sim_time = read_time_axis(sim_data, :sim)
    validate_time_axis(real_time, "real")
    validate_time_axis(sim_time, "simulation")

    targets = [cid for cid in TARGET_CONTACTS
               if haskey_flexible(real_superpulses, cid) &&
                  haskey_flexible(sim_superpulses, cid)]
    isempty(targets) && error("No common collecting-contact superpulses found.")

    real_sets = Dict{Int,Matrix{Float64}}()
    sim_sets = Dict{Int,Matrix{Float64}}()
    global_shifts = Dict{Int,Float64}()

    println("Loading, baseline-subtracting, polarity-correcting, and aligning datasets...")
    for target in targets
        real_wf = Matrix{Float64}(get_flexible(real_superpulses, target))
        sim_wf = Matrix{Float64}(get_flexible(sim_superpulses, target))
        validate_matrix(real_wf, real_time, "real target $target")
        validate_matrix(sim_wf, sim_time, "sim target $target")
        baseline_subtract!(real_wf, real_time, REAL_BASELINE_END_NS)
        baseline_subtract!(sim_wf, sim_time, SIM_BASELINE_END_NS)

        if FLIP_SIM_CORE_POLARITY
            sim_wf[cid_to_index(CORE_CONTACT_ID), :] .*= -1
        end

        shift0 = estimate_global_shift_xcorr(
            real_wf[cid_to_index(ALIGN_REFERENCE_CONTACT), :], real_time,
            sim_wf[cid_to_index(ALIGN_REFERENCE_CONTACT), :], sim_time;
            min_shift_ns=XC_ORIGINAL_MIN_SHIFT_NS,
            max_shift_ns=XC_ORIGINAL_MAX_SHIFT_NS,
            step_ns=XC_ORIGINAL_STEP_NS)

        shift = refine_target_shift(real_wf, sim_wf, real_time, sim_time,
                                    shift0, target;
                                    search_ns=REFINE_SHIFT_HALF_WINDOW_NS,
                                    step_ns=REFINE_SHIFT_STEP_NS)
        real_sets[target] = real_wf
        sim_sets[target] = sim_wf
        global_shifts[target] = shift
        @printf("target %d initial xcorr shift: %.3f ns, refined shift: %.3f ns\n",
                target, shift0, shift)
    end

    if NORMALIZE_SIM_AMPLITUDES
        println("\nApplying per-target simulation amplitude normalization...")
        normalize_sim_amplitudes!(real_sets, sim_sets, real_time, sim_time,
                                  global_shifts, targets)
    end

    println("\nFitting shared outer-contact shaping...")
    outer_tau = fit_shared_outer_tau(real_sets, sim_sets, global_shifts,
                                     real_time, sim_time, targets)
    println("Fitting shared core shaping and core delay...")
    core_tau, core_delay = fit_shared_core_parameters(
        real_sets, sim_sets, global_shifts, real_time, sim_time, targets)

    println("\nShared shaping parameters:")
    @printf("  outer tau:         %.3f ns\n", outer_tau)
    @printf("  core tau:          %.3f ns\n", core_tau)
    @printf("  core extra delay:  %.3f ns\n", core_delay)

    channel_delays = zeros(Float64, NCONTACTS)
    per_target_channel_delays = Dict(
        target => zeros(Float64, NCONTACTS) for target in targets
    )
    if FIT_CHANNEL_DELAYS
        println("\nFitting residual per-channel delays after shared shaping...")
        channel_delays, per_target_channel_delays = fit_shared_channel_delays(
            real_sets, sim_sets, global_shifts, real_time, sim_time, targets,
            outer_tau, core_tau, core_delay)
    end
    report_channel_delays(channel_delays, per_target_channel_delays, targets)

    processed_sim = Dict{Int,Matrix{Float64}}()
    raw_aligned_sim = Dict{Int,Matrix{Float64}}()
    for target in targets
        raw_aligned_sim[target] = prepare_sim_matrix(
            sim_sets[target], sim_time, real_time, global_shifts[target],
            0.0, 0.0, 0.0, zeros(NCONTACTS))
        processed_sim[target] = prepare_sim_matrix(
            sim_sets[target], sim_time, real_time, global_shifts[target],
            outer_tau, core_tau, core_delay, channel_delays)
    end

    println("\nFitting hierarchical crosstalk model...")
    C_shared, D_shared, C_sets, D_sets, deltaC_sets, deltaD_sets =
        fit_hierarchical_crosstalk(
            real_sets, processed_sim, real_time, targets)

    println("\nShared electronics matrices:")
    report_matrices(C_shared, D_shared)
    report_target_corrections(deltaC_sets, deltaD_sets, targets)

    fitted_sets = Dict{Int,Matrix{Float64}}()
    residual_sets = Dict{Int,Matrix{Float64}}()
    metric_rows = NamedTuple[]

    for target in targets
        S = processed_sim[target]
        dS = derivative_matrix(S, median(diff(real_time)))
        C_target = C_sets[target]
        D_target = D_sets[target]
        fitted = C_target * S + D_target * dS
        residual = real_sets[target] - fitted
        fitted_sets[target] = fitted
        residual_sets[target] = residual
        idx = fit_indices(real_time)
        eidx = edge_indices(real_time)

        for cid in ALL_CONTACT_IDS
            i = cid_to_index(cid)
            rmse = sqrt(mean(residual[i, idx].^2))
            scale = max(maximum(abs.(real_sets[target][i, idx])), MIN_CHANNEL_SCALE_KEV)
            corr = safe_cor(real_sets[target][i, idx], fitted[i, idx])
            edge_rmse = NaN
            edge_normalized_rmse = NaN
            if !isempty(eidx)
                edge_rmse = sqrt(mean(residual[i, eidx].^2))
                edge_scale = max(maximum(abs.(real_sets[target][i, eidx])),
                                 MIN_CHANNEL_SCALE_KEV)
                edge_normalized_rmse = edge_rmse / edge_scale
            end
            push!(metric_rows, (
                target_contact=target, channel_contact=cid, rmse_keV=rmse,
                normalized_rmse=rmse/scale, edge_rmse_keV=edge_rmse,
                edge_normalized_rmse=edge_normalized_rmse, correlation=corr))
        end
        plot_fit(real_time, real_sets[target], raw_aligned_sim[target], fitted, target)
        plot_residuals(real_time, residual, target)
    end

    write_matrix_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_shared_integral_crosstalk.csv"),
        C_shared, "dimensionless")
    write_matrix_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_shared_differential_crosstalk_ns.csv"),
        D_shared, "ns")
    write_target_matrix_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_target_integral_crosstalk.csv"),
        C_sets, targets, "dimensionless")
    write_target_matrix_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_target_differential_crosstalk_ns.csv"),
        D_sets, targets, "ns")
    write_target_matrix_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_target_integral_corrections.csv"),
        deltaC_sets, targets, "dimensionless")
    write_target_matrix_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_target_differential_corrections_ns.csv"),
        deltaD_sets, targets, "ns")
    write_metrics_csv(joinpath(OUT_DIR, "$(OUT_PREFIX)_metrics.csv"), metric_rows)
    write_shaping_csv(joinpath(OUT_DIR, "$(OUT_PREFIX)_shaping.csv"),
                      outer_tau, core_tau, core_delay, global_shifts)
    write_channel_delay_csv(
        joinpath(OUT_DIR, "$(OUT_PREFIX)_channel_delays.csv"),
        channel_delays, per_target_channel_delays, targets)

    serialize(joinpath(OUT_DIR, "$(OUT_PREFIX).jls"), Dict(
        "real"=>real_sets,
        "sim_raw_aligned"=>raw_aligned_sim,
        "sim_shaped"=>processed_sim,
        "sim_fitted"=>fitted_sets,
        "residual"=>residual_sets,
        "shared_integral_crosstalk"=>C_shared,
        "shared_differential_crosstalk_ns"=>D_shared,
        "target_integral_crosstalk"=>C_sets,
        "target_differential_crosstalk_ns"=>D_sets,
        "target_integral_corrections"=>deltaC_sets,
        "target_differential_corrections_ns"=>deltaD_sets,
        "time_ns"=>real_time,
        "global_shifts_ns"=>global_shifts,
        "channel_delays_ns"=>channel_delays,
        "per_target_channel_delays_ns"=>per_target_channel_delays,
        "outer_tau_ns"=>outer_tau,
        "core_tau_ns"=>core_tau,
        "core_delay_ns"=>core_delay,
        "metadata"=>Dict(
            "real_file"=>REAL_FILE,
            "sim_file"=>SIM_FILE,
            "flip_sim_core_polarity"=>FLIP_SIM_CORE_POLARITY,
            "zero_offset_enforced"=>true,
            "joint_targets"=>targets,
            "fit_window_ns"=>(FIT_START_NS, FIT_END_NS),
            "edge_window_ns"=>(EDGE_START_NS, EDGE_END_NS),
            "normalize_sim_amplitudes"=>NORMALIZE_SIM_AMPLITUDES,
            "normalize_receiver_channels"=>NORMALIZE_RECEIVER_CHANNELS,
            "two_pole_lowpass"=>true,
            "robust_crosstalk_fit"=>ROBUST_CROSSTALK_FIT,
            "robust_iterations"=>ROBUST_IRLS_ITERATIONS,
            "fit_channel_delays"=>FIT_CHANNEL_DELAYS,
            "fix_core_residual_delay"=>FIX_CORE_RESIDUAL_DELAY,
            "channel_delay_search_ns"=>(CHANNEL_DELAY_MIN_NS,
                                         CHANNEL_DELAY_MAX_NS),
            "channel_delay_fine_step_ns"=>CHANNEL_DELAY_FINE_STEP_NS,
            "alignment_uses_derivative_correlation"=>true,
            "joint_channel_delay_weighting"=>"derivative_energy",
            "derivative_reference_ns"=>DERIVATIVE_REFERENCE_NS,
            "hierarchical_crosstalk"=>true,
            "fit_target_specific_corrections"=>FIT_TARGET_SPECIFIC_CORRECTIONS,
            "target_ridge_integral_collecting"=>TARGET_RIDGE_INTEGRAL_COLLECTING,
            "target_ridge_integral_core"=>TARGET_RIDGE_INTEGRAL_CORE,
            "target_ridge_integral_mirror"=>TARGET_RIDGE_INTEGRAL_MIRROR,
            "target_ridge_differential"=>TARGET_RIDGE_DIFFERENTIAL)))

    println("\nSaved joint fit outputs to $OUT_DIR")
end

# ============================ I/O helpers ============================

function get_required(d, key)
    haskey(d, key) && return d[key]
    haskey(d, Symbol(key)) && return d[Symbol(key)]
    error("Missing key '$key'. Available: $(collect(keys(d)))")
end
haskey_flexible(d, key) = haskey(d,key) || haskey(d,string(key)) ||
                          haskey(d,Symbol(string(key)))
function get_flexible(d, key)
    haskey(d,key) && return d[key]
    haskey(d,string(key)) && return d[string(key)]
    haskey(d,Symbol(string(key))) && return d[Symbol(string(key))]
    error("Missing contact $key")
end
function read_time_axis(data, which)
    haskey(data,"time_ns") && return Float64.(data["time_ns"])
    haskey(data,:time_ns) && return Float64.(data[:time_ns])
    which == :real && return collect(0:REAL_NSAMPLES-1) .* REAL_SAMPLE_PERIOD_NS
    error("Simulation file must contain time_ns")
end
function validate_time_axis(t,label)
    length(t)>=2 || error("$label time axis too short")
    all(diff(t).>0) || error("$label time axis must increase")
end
function validate_matrix(m,t,label)
    size(m,1)==NCONTACTS || error("$label has $(size(m,1)) channels")
    size(m,2)==length(t) || error("$label sample/time mismatch")
end
function cid_to_index(cid)
    i = findfirst(==(cid), ALL_CONTACT_IDS)
    isnothing(i) && error("Unknown contact ID $cid")
    return i
end
fit_indices(t) = findall(x -> FIT_START_NS <= x <= FIT_END_NS, t)
edge_indices(t) = findall(x -> EDGE_START_NS <= x <= EDGE_END_NS, t)

# ============================ Preprocessing ============================

function baseline_subtract!(wf,t,end_ns)
    idx=findall(x->x<=end_ns,t)
    isempty(idx) && error("Empty baseline window")
    for ch in axes(wf,1)
        wf[ch,:] .-= mean(wf[ch,idx])
    end
    wf
end
function crossing_time(y,t,fraction)
    base=mean(y[1:min(5,length(y))]); pk=argmax(abs.(y.-base)); amp=y[pk]-base
    abs(amp)<1e-12 && return NaN
    threshold=base+fraction*amp
    for i in 2:pk
        crossed=amp>0 ? y[i]>=threshold : y[i]<=threshold
        if crossed
            dy=y[i]-y[i-1]; abs(dy)<1e-12 && return t[i]
            f=(threshold-y[i-1])/dy
            return t[i-1]+f*(t[i]-t[i-1])
        end
    end
    NaN
end
function estimate_global_shift(real_y,real_t,sim_y,sim_t,fraction)
    tr=crossing_time(real_y,real_t,fraction); ts=crossing_time(sim_y,sim_t,fraction)
    isfinite(tr)&&isfinite(ts) ? tr-ts : 0.0
end
function derivative_trace(y, dt)
    n = length(y)
    d = zeros(Float64, n)

    if n == 1
        return d
    elseif n == 2
        d[1] = (y[2] - y[1]) / dt
        d[2] = d[1]
        return d
    end

    d[1] = (y[2] - y[1]) / dt
    for k in 2:n-1
        d[k] = (y[k+1] - y[k-1]) / (2 * dt)
    end
    d[n] = (y[n] - y[n-1]) / dt
    return d
end

function estimate_global_shift_xcorr(real_y, real_t, sim_y, sim_t;
        min_shift_ns=-400.0, max_shift_ns=400.0, step_ns=1.0)
    idx = fit_indices(real_t)
    dt = median(diff(real_t))
    y = derivative_trace(real_y, dt)[idx]
    y .-= mean(y)

    best_shift = estimate_global_shift(real_y, real_t, sim_y, sim_t,
                                       ALIGN_FRACTION)
    best_score = -Inf

    for shift in min_shift_ns:step_ns:max_shift_ns
        shifted_sim = sample_shifted(sim_t, sim_y, real_t, shift)
        x = derivative_trace(shifted_sim, dt)[idx]
        x .-= mean(x)
        denom = sqrt(sum(abs2, x) * sum(abs2, y))
        score = denom > 1e-12 ? dot(x, y) / denom : -Inf

        if score > best_score
            best_score = score
            best_shift = shift
        end
    end

    tol = 0.51 * step_ns
    if isapprox(best_shift, min_shift_ns; atol=tol) ||
       isapprox(best_shift, max_shift_ns; atol=tol)
        @warn "Global alignment reached search boundary" best_shift_ns=best_shift min_shift_ns=min_shift_ns max_shift_ns=max_shift_ns score=best_score
    end

    return best_shift
end

function refine_target_shift(real_wf,sim_wf,real_t,sim_t,initial_shift,target;
        search_ns=20.0,step_ns=1.0)
    idx=fit_indices(real_t); it=cid_to_index(target); ic=cid_to_index(CORE_CONTACT_ID)
    best_shift=initial_shift; best_loss=Inf
    for shift in (initial_shift-search_ns):step_ns:(initial_shift+search_ns)
        loss=0.0
        for i in (it,ic)
            x=sample_shifted(sim_t,vec(sim_wf[i,:]),real_t,shift)
            y=vec(real_wf[i,:]); gain=zero_offset_gain(x[idx],y[idx])
            scale=max(maximum(abs.(y[idx])),MIN_CHANNEL_SCALE_KEV)
            loss+=mean(((y[idx].-gain.*x[idx])./scale).^2)
        end
        if loss<best_loss; best_loss=loss; best_shift=shift; end
    end
    lower = initial_shift - search_ns
    upper = initial_shift + search_ns
    tol = 0.51 * step_ns
    if isapprox(best_shift, lower; atol=tol) ||
       isapprox(best_shift, upper; atol=tol)
        @warn "Target shift refinement reached search boundary" target=target best_shift_ns=best_shift lower_ns=lower upper_ns=upper
    end
    return best_shift
end
function interp_constant_edges(x,y,xnew)
    out=Vector{Float64}(undef,length(xnew)); j=1
    for i in eachindex(xnew)
        xx=xnew[i]
        if xx<=x[1]; out[i]=y[1]
        elseif xx>=x[end]; out[i]=y[end]
        else
            while j<length(x)-1 && x[j+1]<xx; j+=1; end
            f=(xx-x[j])/(x[j+1]-x[j]); out[i]=y[j]+f*(y[j+1]-y[j])
        end
    end
    out
end
sample_shifted(st,sy,tt,shift)=interp_constant_edges(st,sy,tt.-shift)
function lowpass_onepole(y,dt,tau)
    tau<=0 && return copy(y)
    a=exp(-dt/tau); out=similar(y); out[1]=y[1]
    for k in 2:length(y); out[k]=a*out[k-1]+(1-a)*y[k]; end
    out
end
function lowpass(y,dt,tau)
    tau<=0 && return copy(y)
    lowpass_onepole(lowpass_onepole(y,dt,tau),dt,tau)
end

function prepare_sim_matrix(sim,sim_t,real_t,shift,outer_tau,core_tau,
                            core_delay,channel_delays)
    length(channel_delays)==NCONTACTS ||
        error("channel_delays must have $NCONTACTS entries")
    out=zeros(Float64,NCONTACTS,length(real_t)); dt=median(diff(real_t))
    for cid in ALL_CONTACT_IDS
        i=cid_to_index(cid)
        residual_delay=channel_delays[i]
        local_delay=residual_delay+(cid==CORE_CONTACT_ID ? core_delay : 0.0)
        tau=cid==CORE_CONTACT_ID ? core_tau : outer_tau
        y=sample_shifted(sim_t,vec(sim[i,:]),real_t,shift+local_delay)
        out[i,:].=lowpass(y,dt,tau)
    end
    out
end

function normalize_sim_amplitudes!(real_sets,sim_sets,real_t,sim_t,shifts,targets)
    idx=fit_indices(real_t)
    for target in targets
        i=cid_to_index(target)
        x=sample_shifted(sim_t,vec(sim_sets[target][i,:]),real_t,shifts[target])
        y=vec(real_sets[target][i,:]); gain=zero_offset_gain(x[idx],y[idx])
        if isfinite(gain)&&abs(gain)>1e-12
            sim_sets[target].*=gain
            @printf("target %d simulation amplitude gain applied: %.6f\n",target,gain)
        else
            @warn "Could not compute stable amplitude gain" target=target gain=gain
        end
    end
    sim_sets
end

# ============================ Shared shaping fit ============================

function fit_shared_outer_tau(real_sets,sim_sets,shifts,real_t,sim_t,targets)
    best_tau=0.0; best_loss=Inf
    for tau in OUTER_TAU_MIN_NS:COARSE_STEP_NS:OUTER_TAU_MAX_NS
        loss=target_channel_loss(tau,real_sets,sim_sets,shifts,real_t,sim_t,targets)
        if loss<best_loss; best_loss=loss; best_tau=tau; end
    end
    lo=max(OUTER_TAU_MIN_NS,best_tau-COARSE_STEP_NS)
    hi=min(OUTER_TAU_MAX_NS,best_tau+COARSE_STEP_NS)
    for tau in lo:FINE_STEP_NS:hi
        loss=target_channel_loss(tau,real_sets,sim_sets,shifts,real_t,sim_t,targets)
        if loss<best_loss; best_loss=loss; best_tau=tau; end
    end
    best_tau
end
function target_channel_loss(tau,real_sets,sim_sets,shifts,real_t,sim_t,targets)
    idx=fit_indices(real_t); dt=median(diff(real_t)); total=0.0
    for target in targets
        i=cid_to_index(target)
        x=sample_shifted(sim_t,vec(sim_sets[target][i,:]),real_t,shifts[target])
        x=lowpass(x,dt,tau); y=vec(real_sets[target][i,:])
        gain=zero_offset_gain(x[idx],y[idx]); scale=max(maximum(abs.(y[idx])),MIN_CHANNEL_SCALE_KEV)
        total+=mean(((y[idx].-gain.*x[idx])./scale).^2)
    end
    total/length(targets)
end
function fit_shared_core_parameters(real_sets,sim_sets,shifts,real_t,sim_t,targets)
    best_tau=0.0; best_delay=0.0; best_loss=Inf
    for delay in CORE_DELAY_MIN_NS:COARSE_STEP_NS:CORE_DELAY_MAX_NS,
        tau in CORE_TAU_MIN_NS:COARSE_STEP_NS:CORE_TAU_MAX_NS
        loss=core_channel_loss(tau,delay,real_sets,sim_sets,shifts,real_t,sim_t,targets)
        if loss<best_loss; best_loss=loss; best_tau=tau; best_delay=delay; end
    end
    dlo=max(CORE_DELAY_MIN_NS,best_delay-COARSE_STEP_NS)
    dhi=min(CORE_DELAY_MAX_NS,best_delay+COARSE_STEP_NS)
    tlo=max(CORE_TAU_MIN_NS,best_tau-COARSE_STEP_NS)
    thi=min(CORE_TAU_MAX_NS,best_tau+COARSE_STEP_NS)
    for delay in dlo:FINE_STEP_NS:dhi, tau in tlo:FINE_STEP_NS:thi
        loss=core_channel_loss(tau,delay,real_sets,sim_sets,shifts,real_t,sim_t,targets)
        if loss<best_loss; best_loss=loss; best_tau=tau; best_delay=delay; end
    end
    best_tau,best_delay
end
function core_channel_loss(tau,delay,real_sets,sim_sets,shifts,real_t,sim_t,targets)
    idx=fit_indices(real_t); dt=median(diff(real_t)); i=cid_to_index(CORE_CONTACT_ID); total=0.0
    for target in targets
        x=sample_shifted(sim_t,vec(sim_sets[target][i,:]),real_t,shifts[target]+delay)
        x=lowpass(x,dt,tau); y=vec(real_sets[target][i,:])
        gain=zero_offset_gain(x[idx],y[idx]); scale=max(maximum(abs.(y[idx])),MIN_CHANNEL_SCALE_KEV)
        total+=mean(((y[idx].-gain.*x[idx])./scale).^2)
    end
    total/length(targets)
end
zero_offset_gain(x,y)=dot(x,y)/max(dot(x,x),1e-12)

# ============================ Per-channel delay fit ============================

function shifted_real_grid_trace(y, real_t, delay_ns)
    # Positive delay moves the simulated waveform later.
    return interp_constant_edges(real_t, y, real_t .- delay_ns)
end

function timing_information(y, idx, dt)
    dy = derivative_trace(y, dt)
    return sum(abs2, dy[idx])
end

function fit_delay_on_real_grid(real_trace, base_sim_trace, real_t)
    idx = fit_indices(real_t)
    best_delay = 0.0
    best_loss = Inf

    function loss_at(delay_ns)
        x = shifted_real_grid_trace(base_sim_trace, real_t, delay_ns)
        gain = zero_offset_gain(x[idx], real_trace[idx])
        scale = max(maximum(abs.(real_trace[idx])), MIN_CHANNEL_SCALE_KEV)
        return mean(((real_trace[idx] .- gain .* x[idx]) ./ scale).^2)
    end

    for delay_ns in CHANNEL_DELAY_MIN_NS:CHANNEL_DELAY_COARSE_STEP_NS:CHANNEL_DELAY_MAX_NS
        loss = loss_at(delay_ns)
        if loss < best_loss
            best_loss = loss
            best_delay = delay_ns
        end
    end

    lo = max(CHANNEL_DELAY_MIN_NS,
             best_delay - CHANNEL_DELAY_COARSE_STEP_NS)
    hi = min(CHANNEL_DELAY_MAX_NS,
             best_delay + CHANNEL_DELAY_COARSE_STEP_NS)

    for delay_ns in lo:CHANNEL_DELAY_FINE_STEP_NS:hi
        loss = loss_at(delay_ns)
        if loss < best_loss
            best_loss = loss
            best_delay = delay_ns
        end
    end

    return best_delay, best_loss
end

function shared_channel_delay_loss(delay_ns, receiver_index, real_sets,
                                   base_sim_sets, real_t, targets)
    idx = fit_indices(real_t)
    dt = median(diff(real_t))
    weighted_loss = 0.0
    total_weight = 0.0

    for target in targets
        y = vec(real_sets[target][receiver_index, :])
        x0 = vec(base_sim_sets[target][receiver_index, :])
        x = shifted_real_grid_trace(x0, real_t, delay_ns)
        gain = zero_offset_gain(x[idx], y[idx])
        scale = max(maximum(abs.(y[idx])), MIN_CHANNEL_SCALE_KEV)
        dataset_loss = mean(((y[idx] .- gain .* x[idx]) ./ scale).^2)

        real_info = timing_information(y, idx, dt)
        sim_info = timing_information(x0, idx, dt)
        weight = sqrt(max(real_info, 0.0) * max(sim_info, 0.0))

        if isfinite(dataset_loss) && isfinite(weight) && weight > 1e-12
            weighted_loss += weight * dataset_loss
            total_weight += weight
        end
    end

    return total_weight > 1e-12 ? weighted_loss / total_weight : Inf
end

function fit_one_shared_channel_delay(receiver_index, real_sets, base_sim_sets,
                                      real_t, targets)
    best_delay = 0.0
    best_loss = Inf

    for delay_ns in CHANNEL_DELAY_MIN_NS:CHANNEL_DELAY_COARSE_STEP_NS:CHANNEL_DELAY_MAX_NS
        loss = shared_channel_delay_loss(delay_ns, receiver_index, real_sets,
                                         base_sim_sets, real_t, targets)
        if loss < best_loss
            best_loss = loss
            best_delay = delay_ns
        end
    end

    lo = max(CHANNEL_DELAY_MIN_NS,
             best_delay - CHANNEL_DELAY_COARSE_STEP_NS)
    hi = min(CHANNEL_DELAY_MAX_NS,
             best_delay + CHANNEL_DELAY_COARSE_STEP_NS)

    for delay_ns in lo:CHANNEL_DELAY_FINE_STEP_NS:hi
        loss = shared_channel_delay_loss(delay_ns, receiver_index, real_sets,
                                         base_sim_sets, real_t, targets)
        if loss < best_loss
            best_loss = loss
            best_delay = delay_ns
        end
    end

    tol = 0.51 * CHANNEL_DELAY_FINE_STEP_NS
    if isapprox(best_delay, CHANNEL_DELAY_MIN_NS; atol=tol) ||
       isapprox(best_delay, CHANNEL_DELAY_MAX_NS; atol=tol)
        @warn "Channel delay reached search boundary"             channel_index=receiver_index delay_ns=best_delay loss=best_loss
    end

    return best_delay, best_loss
end

function fit_shared_channel_delays(real_sets, sim_sets, shifts, real_t, sim_t,
                                   targets, outer_tau, core_tau, core_delay)
    zero_delays = zeros(Float64, NCONTACTS)
    base_sim_sets = Dict{Int,Matrix{Float64}}()

    for target in targets
        base_sim_sets[target] = prepare_sim_matrix(
            sim_sets[target], sim_t, real_t, shifts[target], outer_tau,
            core_tau, core_delay, zero_delays)
    end

    shared_delays = zeros(Float64, NCONTACTS)
    per_target_delays = Dict(
        target => fill(NaN, NCONTACTS) for target in targets)

    for cid in ALL_CONTACT_IDS
        i = cid_to_index(cid)

        if cid == CORE_CONTACT_ID && FIX_CORE_RESIDUAL_DELAY
            shared_delays[i] = 0.0
            for target in targets
                per_target_delays[target][i] = 0.0
            end
            continue
        end

        delay, loss = fit_one_shared_channel_delay(
            i, real_sets, base_sim_sets, real_t, targets)
        shared_delays[i] = delay
        @printf("channel %d joint delay = %.3f ns, loss = %.6e\n",
                cid, delay, loss)

        # Diagnostic only. Final delays come from the joint weighted fit above.
        for target in targets
            target_delay, _ = fit_delay_on_real_grid(
                vec(real_sets[target][i, :]),
                vec(base_sim_sets[target][i, :]), real_t)
            per_target_delays[target][i] = target_delay
        end
    end

    return shared_delays, per_target_delays
end

# ============================ Joint crosstalk fit ============================

function derivative_matrix(S,dt)
    d=zeros(size(S)); ns=size(S,2)
    for ch in axes(S,1)
        if ns==1; d[ch,1]=0.0
        elseif ns==2
            d[ch,1]=(S[ch,2]-S[ch,1])/dt; d[ch,2]=d[ch,1]
        else
            d[ch,1]=(S[ch,2]-S[ch,1])/dt
            for k in 2:ns-1; d[ch,k]=(S[ch,k+1]-S[ch,k-1])/(2dt); end
            d[ch,ns]=(S[ch,ns]-S[ch,ns-1])/dt
        end
    end
    d
end
function crosstalk_lambdas(receiver)
    lambda_C=Float64[]; lambda_D=Float64[]
    for source in ALL_CONTACT_IDS
        if source==receiver; push!(lambda_C,RIDGE_DIAGONAL)
        elseif source==CORE_CONTACT_ID; push!(lambda_C,RIDGE_CORE_TO_CONTACT)
        elseif receiver==CORE_CONTACT_ID; push!(lambda_C,RIDGE_CONTACT_TO_CORE)
        else; push!(lambda_C,RIDGE_CONTACT_TO_CONTACT)
        end
    end
    for source in ALL_CONTACT_IDS
        push!(lambda_D,source==receiver ? RIDGE_DIFFERENTIAL_DIAGONAL :
                                         RIDGE_DIFFERENTIAL_OFFDIAGONAL)
    end
    vcat(lambda_C,lambda_D)
end
function fit_shared_crosstalk(real_sets,sim_sets,real_t,targets)
    idx=fit_indices(real_t); dt=median(diff(real_t))
    C=zeros(Float64,NCONTACTS,NCONTACTS); D=zeros(Float64,NCONTACTS,NCONTACTS)
    for receiver in ALL_CONTACT_IDS
        rows=Vector{Vector{Float64}}(); ys=Float64[]; ri=cid_to_index(receiver)
        scale=1.0
        if NORMALIZE_RECEIVER_CHANNELS
            allvals=vcat([vec(real_sets[t][ri,idx]) for t in targets]...)
            scale=max(maximum(abs.(allvals)),MIN_CHANNEL_SCALE_KEV)
        end
        for target in targets
            S=sim_sets[target]; dS=derivative_matrix(S,dt)
            for k in idx
                push!(rows, vcat(S[:, k],
                                  DERIVATIVE_REFERENCE_NS .* dS[:, k]) ./ scale)
                push!(ys,real_sets[target][ri,k]/scale)
            end
        end
        X=reduce(vcat,[permutedims(r) for r in rows]); y=ys
        beta0=zeros(2NCONTACTS)
        same_sim=vcat([vec(sim_sets[t][ri,idx]) for t in targets]...)
        same_real=vcat([vec(real_sets[t][ri,idx]) for t in targets]...)
        beta0[ri]=dot(same_sim,same_real)>=0 ? 1.0 : -1.0
        lambdas=crosstalk_lambdas(receiver)
        beta=ROBUST_CROSSTALK_FIT ? robust_ridge_solve(X,y,beta0,lambdas) :
                                   ridge_solve(X,y,beta0,lambdas)
        C[ri,:].=beta[1:NCONTACTS]
        D[ri, :] .= DERIVATIVE_REFERENCE_NS .*                     beta[NCONTACTS+1:2NCONTACTS]
    end
    C,D
end
function target_correction_lambdas(receiver, target)
    lambda_C = zeros(Float64, NCONTACTS)
    lambda_D = fill(TARGET_RIDGE_DIFFERENTIAL, NCONTACTS)

    for source in ALL_CONTACT_IDS
        j = cid_to_index(source)
        if receiver == target && source == target
            lambda_C[j] = TARGET_RIDGE_INTEGRAL_COLLECTING
        elseif receiver == CORE_CONTACT_ID && source == CORE_CONTACT_ID
            lambda_C[j] = TARGET_RIDGE_INTEGRAL_CORE
        else
            lambda_C[j] = TARGET_RIDGE_INTEGRAL_MIRROR
        end
    end

    return vcat(lambda_C, lambda_D)
end

function fit_target_correction(real_wf, sim_wf, real_t, target,
                               C_shared, D_shared)
    idx = fit_indices(real_t)
    dt = median(diff(real_t))
    dS = derivative_matrix(sim_wf, dt)
    shared_fit = C_shared * sim_wf + D_shared * dS
    residual = real_wf - shared_fit

    deltaC = zeros(Float64, NCONTACTS, NCONTACTS)
    deltaD = zeros(Float64, NCONTACTS, NCONTACTS)

    for receiver in ALL_CONTACT_IDS
        ri = cid_to_index(receiver)
        scale = max(maximum(abs.(real_wf[ri, idx])), MIN_CHANNEL_SCALE_KEV)

        X = zeros(Float64, length(idx), 2 * NCONTACTS)
        y = zeros(Float64, length(idx))

        for (row, k) in enumerate(idx)
            X[row, 1:NCONTACTS] .= sim_wf[:, k] ./ scale
            X[row, NCONTACTS+1:2NCONTACTS] .=
                DERIVATIVE_REFERENCE_NS .* dS[:, k] ./ scale
            y[row] = residual[ri, k] / scale
        end

        beta0 = zeros(Float64, 2 * NCONTACTS)
        lambdas = target_correction_lambdas(receiver, target)
        beta = robust_ridge_solve_iterations(
            X, y, beta0, lambdas, TARGET_ROBUST_IRLS_ITERATIONS)

        deltaC[ri, :] .= beta[1:NCONTACTS]
        deltaD[ri, :] .= DERIVATIVE_REFERENCE_NS .*
                         beta[NCONTACTS+1:2NCONTACTS]
    end

    return deltaC, deltaD
end

function fit_hierarchical_crosstalk(real_sets, sim_sets, real_t, targets)
    # Stage 1: common electronics response across all collecting-contact sets.
    C_shared, D_shared = fit_shared_crosstalk(
        real_sets, sim_sets, real_t, targets)

    C_sets = Dict{Int,Matrix{Float64}}()
    D_sets = Dict{Int,Matrix{Float64}}()
    deltaC_sets = Dict{Int,Matrix{Float64}}()
    deltaD_sets = Dict{Int,Matrix{Float64}}()

    for target in targets
        if FIT_TARGET_SPECIFIC_CORRECTIONS
            deltaC, deltaD = fit_target_correction(
                real_sets[target], sim_sets[target], real_t, target,
                C_shared, D_shared)
        else
            deltaC = zeros(Float64, NCONTACTS, NCONTACTS)
            deltaD = zeros(Float64, NCONTACTS, NCONTACTS)
        end

        deltaC_sets[target] = deltaC
        deltaD_sets[target] = deltaD
        C_sets[target] = C_shared + deltaC
        D_sets[target] = D_shared + deltaD
    end

    return C_shared, D_shared, C_sets, D_sets,
           deltaC_sets, deltaD_sets
end

function robust_ridge_solve_iterations(X, y, beta0, lambdas, iterations)
    beta = copy(beta0)
    weights = ones(length(y))

    for _ in 1:iterations
        ws = sqrt.(weights)
        Xw = X .* reshape(ws, :, 1)
        yw = y .* ws
        A = Xw' * Xw + Diagonal(lambdas)
        b = Xw' * yw + lambdas .* beta0
        beta = A \ b

        r = y .- X * beta
        medr = median(r)
        sigma = 1.4826 * median(abs.(r .- medr)) + 1e-12
        cutoff = HUBER_K * sigma
        weights .= ifelse.(abs.(r) .<= cutoff, 1.0,
                           cutoff ./ max.(abs.(r), 1e-12))
    end

    return beta
end

function ridge_solve(X,y,beta0,lambdas)
    (X'*X+Diagonal(lambdas))\(X'*y+lambdas.*beta0)
end
function robust_ridge_solve(X,y,beta0,lambdas)
    beta=copy(beta0); weights=ones(length(y))
    for _ in 1:ROBUST_IRLS_ITERATIONS
        ws=sqrt.(weights); Xw=X.*reshape(ws,:,1); yw=y.*ws
        beta=(Xw'*Xw+Diagonal(lambdas))\(Xw'*yw+lambdas.*beta0)
        r=y.-X*beta; medr=median(r)
        sigma=1.4826*median(abs.(r.-medr))+1e-12; c=HUBER_K*sigma
        weights.=ifelse.(abs.(r).<=c,1.0,c./max.(abs.(r),1e-12))
    end
    beta
end
safe_cor(a,b)=(std(a)<1e-12||std(b)<1e-12) ? NaN : cor(a,b)

# ============================ Reports and outputs ============================

function report_channel_delays(shared,per_target,targets)
    println("\nResidual channel delays:")
    println("  positive delay = simulated waveform moved later")
    for cid in ALL_CONTACT_IDS
        i=cid_to_index(cid)
        vals=haskey(per_target,first(targets)) ? [per_target[t][i] for t in targets] : Float64[]
        spread=isempty(vals) ? NaN : (length(vals)>1 ? std(vals) : 0.0)
        @printf("  channel %d: shared = %8.3f ns",cid,shared[i])
        isfinite(spread) && @printf(", target-to-target std = %.3f ns",spread)
        println()
        abs(shared[i])>WARN_CHANNEL_DELAY_NS_ABS &&
            @warn "Large residual channel delay" contact=cid delay_ns=shared[i]
    end
end
function report_matrices(C,D)
    println("\nIntegral matrix C:"); display(C)
    println("Differential matrix D [ns]:"); display(D)
    for i in 1:NCONTACTS,j in 1:NCONTACTS
        i!=j && abs(C[i,j])>WARN_INTEGRAL_CROSSTALK_ABS &&
            @warn "Large integral crosstalk" receiver=i source=j coefficient=C[i,j]
        abs(D[i,j])>WARN_DIFFERENTIAL_CROSSTALK_NS_ABS &&
            @warn "Large differential crosstalk" receiver=i source=j coefficient_ns=D[i,j]
    end
end
function report_target_corrections(deltaC_sets, deltaD_sets, targets)
    println("\nTarget-specific correction summary:")
    for target in sort(targets)
        dC = deltaC_sets[target]
        dD = deltaD_sets[target]
        @printf("  target %d: max |deltaC| = %.4f, max |deltaD| = %.4f ns\n",
                target, maximum(abs.(dC)), maximum(abs.(dD)))

        maximum(abs.(dC)) > WARN_TARGET_INTEGRAL_CORRECTION_ABS &&
            @warn "Large target-specific integral correction"                 target=target maximum_abs=maximum(abs.(dC))
        maximum(abs.(dD)) > WARN_TARGET_DIFFERENTIAL_CORRECTION_NS_ABS &&
            @warn "Large target-specific differential correction"                 target=target maximum_abs_ns=maximum(abs.(dD))
    end
end

function write_target_matrix_csv(path, matrices, targets, unit)
    open(path, "w") do io
        println(io, "target_contact,receiver_contact,source_contact,value,unit")
        for target in sort(targets)
            M = matrices[target]
            for i in 1:NCONTACTS, j in 1:NCONTACTS
                println(io,
                    "$target,$(ALL_CONTACT_IDS[i]),$(ALL_CONTACT_IDS[j]),$(M[i,j]),$unit")
            end
        end
    end
end

function write_matrix_csv(path,M,unit)
    open(path,"w") do io
        println(io,"receiver_contact,source_contact,value,unit")
        for i in 1:NCONTACTS,j in 1:NCONTACTS
            println(io,"$(ALL_CONTACT_IDS[i]),$(ALL_CONTACT_IDS[j]),$(M[i,j]),$unit")
        end
    end
end
function write_metrics_csv(path,rows)
    open(path,"w") do io
        println(io,"target_contact,channel_contact,rmse_keV,normalized_rmse,edge_rmse_keV,edge_normalized_rmse,correlation")
        for r in rows; println(io,join(Tuple(r),",")); end
    end
end
function write_shaping_csv(path,outer_tau,core_tau,core_delay,shifts)
    open(path,"w") do io
        println(io,"parameter,target_contact,value_ns")
        println(io,"outer_tau,all,$outer_tau")
        println(io,"core_tau,all,$core_tau")
        println(io,"core_extra_delay,all,$core_delay")
        for t in sort(collect(keys(shifts))); println(io,"global_shift,$t,$(shifts[t])"); end
    end
end
function write_channel_delay_csv(path,shared,per_target,targets)
    open(path,"w") do io
        println(io,"scope,target_contact,channel_contact,delay_ns")
        for cid in ALL_CONTACT_IDS
            i=cid_to_index(cid); println(io,"shared,all,$cid,$(shared[i])")
        end
        for target in sort(targets),cid in ALL_CONTACT_IDS
            i=cid_to_index(cid)
            println(io,"per_target,$target,$cid,$(per_target[target][i])")
        end
    end
end
function plot_fit(t,real,raw,fitted,target)
    p=plot(layout=(3,3),size=(1500,1100),plot_title="Target contact $target: joint electronics fit")
    for cid in ALL_CONTACT_IDS
        i=cid_to_index(cid)
        plot!(p[i],t,real[i,:],label="real",linewidth=2.5,color=:black)
        plot!(p[i],t,raw[i,:],label="sim aligned",linewidth=1.2,linestyle=:dash,color=:blue)
        plot!(p[i],t,fitted[i,:],label="joint fitted",linewidth=2,color=:red)
        title!(p[i],cid==CORE_CONTACT_ID ? "FV/core" : "contact $cid")
        xlabel!(p[i],"time / ns"); ylabel!(p[i],"signal / keV")
        xlims!(p[i],FIT_START_NS,FIT_END_NS)
    end
    savefig(p,joinpath(OUT_DIR,"$(OUT_PREFIX)_target$(target).png"))
end
function plot_residuals(t,residual,target)
    p=plot(layout=(3,3),size=(1500,1100),plot_title="Target contact $target: residuals")
    for cid in ALL_CONTACT_IDS
        i=cid_to_index(cid)
        plot!(p[i],t,residual[i,:],label="real - fit",linewidth=1.5,color=:purple)
        hline!(p[i],[0.0],label="",color=:black,linewidth=0.8)
        title!(p[i],cid==CORE_CONTACT_ID ? "FV/core" : "contact $cid")
        xlabel!(p[i],"time / ns"); ylabel!(p[i],"residual / keV")
        xlims!(p[i],FIT_START_NS,FIT_END_NS)
    end
    savefig(p,joinpath(OUT_DIR,"$(OUT_PREFIX)_target$(target)_residuals.png"))
end

main()