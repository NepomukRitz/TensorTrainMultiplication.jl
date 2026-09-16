"""
    MultiplyInfo

What one [`multiply`](@ref) call did. `peak_bonddim` is the largest bond dimension that
existed at any moment: intermediate bonds and, while they are still part of the chain, the
inputs' own bonds, so it is never below the larger input rank and `maxbonddim` bounds only the
bonds the algorithm creates. `peak_bonddim_per_step[k]` is the same for step `k` alone, where
step `k` contracts carriage `N + 1 - k`; `bonddims` the bonds of the returned train;
`n_swaps = N (N - 1) / 2`; `discarded_weight` the sum over every truncation of the relative
discarded squared weight, noise-floor removals included; `hit_maxbonddim` whether
`maxbonddim` removed anything.

Accuracy, three fields that mean different things:

- `error_estimate` and `error_stderr`: the sampled relative L2 error
  `||C - A o B||_2 / ||A o B||_2` of the returned product and its standard error, from
  `nsamples` uniform random multi-indices against the exact `A(x) B(x)`; consistent, no
  reference computation. `NaN` when `nsamples == 0`.
- `verified`: a `tolerance` was requested, `error_estimate + 2 error_stderr <= tolerance`,
  and the sample saw the residual (see `min_ess`). `false` in `cutoff` mode. It says that the
  *sampled* residual is below the tolerance at this sample size, not that the true error is;
  a residual living on a vanishing fraction of the grid can pass it.
- `error_bound`: the rigorous bound `sum_i sqrt(w_i)` over every truncation (each happens in
  mixed-canonical gauge, so its error is exactly `sqrt(w_i)` times the current norm, and the
  triangle inequality sums them). Always valid, and pessimistic by a factor that grows with
  the number of truncating swaps: 1.9-17.4 on 12-site products, about 8 on a 28-site production
  product. A diagnostic, not the control.

`attempts` counts the passes run and `cutoffs` lists their cutoffs, the last being the
returned product's.

Three fields describe the sample itself, all `NaN` when nothing was sampled:

- `ess_num`: the effective sample size of the residual (see [`effective_sample_size`](@ref)) --
  how much of the sample carried the error. This is the one `verified` gates on.
- `ess_den`: the same for the product's own magnitude. Reported, not gated on: a sharply
  peaked product is heavy-tailed here by construction, and a small `ess_den` inflates the
  uncertainty of the estimate by about `1 / sqrt(ess_den)` in relative terms rather than
  invalidating it.
- `min_ess`: the `ess_num` that `verified` demands, unless `error_bound` already meets the
  tolerance on its own.
"""
struct MultiplyInfo
    peak_bonddim::Int
    peak_bonddim_per_step::Vector{Int}
    bonddims::Vector{Int}
    n_swaps::Int
    discarded_weight::Float64
    hit_maxbonddim::Bool
    error_bound::Float64
    error_estimate::Float64
    error_stderr::Float64
    verified::Bool
    attempts::Int
    cutoffs::Vector{Float64}
    nsamples::Int
    ess_num::Float64
    ess_den::Float64
    min_ess::Int
end

function _check_inputs(A, B)
    N = length(A)
    N == length(B) || throw(DimensionMismatch("A has $(N) cores, B has $(length(B))"))
    N >= 1 || throw(ArgumentError("multiply needs at least one core"))
    for i in 1:N
        size(A[i], 2) == size(B[i], 2) || throw(DimensionMismatch(
            "site dimension at core $(i): A has $(size(A[i], 2)), B has $(size(B[i], 2))"))
    end
    for (name, X) in (("A", A), ("B", B))
        size(X[1], 1) == 1 || throw(ArgumentError(
            "$(name): the first core's left bond must have dimension 1, got $(size(X[1], 1))"))
        size(X[N], 3) == 1 || throw(ArgumentError(
            "$(name): the last core's right bond must have dimension 1, got $(size(X[N], 3))"))
        for i in 1:(N - 1)
            size(X[i], 3) == size(X[i + 1], 1) || throw(DimensionMismatch(
                "$(name): bond between cores $(i) and $(i + 1) has dimensions $(size(X[i], 3)) and $(size(X[i + 1], 1))"))
        end
    end
    return N
end

"""
    _multiply_pass(A, B, cutoff, maxbonddim, final_truncation) -> (cores, pass)

One swap-network pass at a fixed `cutoff`, returning the product's cores and a NamedTuple
carrying the pass's own diagnostics, named as in [`MultiplyInfo`](@ref).
"""
function _multiply_pass(A::AbstractVector{<:AbstractArray{TA,3}}, B::AbstractVector{<:AbstractArray{TB,3}},
                        cutoff::Real, maxbonddim::Integer,
                        final_truncation::Bool) where {TA<:Number,TB<:Number}
    T = promote_type(TA, TB)
    N = _check_inputs(A, B)
    if N == 1
        d = size(A[1], 2)
        C = Array{T,3}(undef, 1, d, 1)
        for s in 1:d
            C[1, s, 1] = A[1][1, s, 1] * B[1][1, s, 1]
        end
        return ([C], (peak_bonddim = 1, peak_bonddim_per_step = [1], bonddims = Int[],
                      n_swaps = 0, discarded_weight = 0.0, hit_maxbonddim = false,
                      error_bound = 0.0))
    end

    f = left_canonical!([Array{T,3}(a) for a in A])
    g = left_canonical!([Array{T,3}(b) for b in B])
    # Chain [f_1 .. f_N, g_N .. g_1]: g reversed, each g core flipped so its legs read
    # (left, site, right) in chain orientation. The boundary bonds of f_N and g_N have
    # dimension one and already form the chain's link; f_1 .. f_{N-1} are left-orthogonal,
    # the flipped g_1 .. g_{N-1} are right-orthogonal, the norm sits on the pair (f_N, g_N).
    chain = Vector{Array{T,3}}(undef, 2N)
    for j in 1:N
        chain[j] = f[j]
        chain[N + j] = permutedims(g[N + 1 - j], (3, 2, 1))
    end
    len = 2N

    peak_per_step = zeros(Int, N)
    discarded = 0.0
    bound = 0.0
    capped = false
    for i in N:-1:1
        # chain[1:len] = [f_1 .. f_i, C_{i+1} .. C_N, g_i, g_{i-1} .. g_1]; the centre is at
        # position i+1 (position N when i == N).
        # 1. QR gauge sweep: move the centre from position i+1 onto position N, so that each
        #    swap below is a global truncation (everything left of the swapped pair is
        #    left-orthogonal, everything right of it right-orthogonal).
        for p in (i + 1):(N - 1)
            l, d, r = size(chain[p])
            F = qr!(reshape(chain[p], l * d, r))
            Q = Matrix(F.Q)
            k = size(Q, 2)
            chain[p] = reshape(Q, l, d, k)
            l2, d2, r2 = size(chain[p + 1])
            chain[p + 1] = reshape(F.R * reshape(chain[p + 1], l2, d2 * r2), k, d2, r2)
        end
        # 2. Swap g_i from position N+1 down to position i+1, one truncated SVD per swap. The
        #    left factor keeps the left bond and takes g_i's site; the right factor takes the
        #    product carriage's site and the right bond.
        step_peak = 0
        for q in (N + 1):-1:(i + 2)
            Cq = chain[q - 1]
            G = chain[q]
            l, dc, m = size(Cq)
            dg, r = size(G, 2), size(G, 3)
            M = reshape(Cq, l * dc, m) * reshape(G, m, dg * r)
            Tm = reshape(permutedims(reshape(M, l, dc, dg, r), (1, 3, 2, 4)), l * dg, dc * r)
            F = svd!(Tm)
            k, w, c = truncated_rank(F.S, cutoff, maxbonddim)
            discarded += w
            bound += sqrt(w)
            capped |= c
            U = F.U[:, 1:k]
            rmul!(U, Diagonal(F.S[1:k]))
            chain[q - 1] = reshape(U, l, dg, k)
            chain[q] = reshape(F.Vt[1:k, :], k, dc, r)
            step_peak = max(step_peak, k)
        end
        # 3. COPY contraction of f_i (position i) with g_i (position i+1):
        #    C[l, s, r] = sum_m f_i[l, s, m] g_i[m, s, r], one matrix product per site value.
        Fi = chain[i]
        Gi = chain[i + 1]
        l, d, m = size(Fi)
        r = size(Gi, 3)
        C = Array{T,3}(undef, l, d, r)
        for s in 1:d
            @views mul!(C[:, s, :], Fi[:, s, :], Gi[:, s, :])
        end
        chain[i] = C
        for p in (i + 1):(len - 1)
            chain[p] = chain[p + 1]
        end
        len -= 1
        after = maximum(size(chain[p], 3) for p in 1:(len - 1))
        peak_per_step[N + 1 - i] = max(step_peak, after)
    end

    cores = chain[1:N]                 # centre on C_1, C_2 .. C_N right-orthogonal
    if final_truncation
        w, c, sb = sweep_truncate!(cores, cutoff, maxbonddim)
        discarded += w
        bound += sb
        capped |= c
    end
    pass = (peak_bonddim = maximum(peak_per_step), peak_bonddim_per_step = peak_per_step,
            bonddims = [size(c, 3) for c in cores[1:(N - 1)]], n_swaps = N * (N - 1) ÷ 2,
            discarded_weight = discarded, hit_maxbonddim = capped, error_bound = bound)
    return cores, pass
end

"""
    multiply(A, B; tolerance, maxbonddim = typemax(Int), final_truncation = true,
             nsamples = 4000, rng = Random.default_rng(),
             pilot_cutoff = tolerance^2 / (N - 1)^2, safety = 0.5, max_attempts = 3,
             min_ess = 30) -> (C, info)
    multiply(A, B; cutoff, maxbonddim = typemax(Int), final_truncation = true,
             nsamples = 0, rng = Random.default_rng()) -> (C, info)

Elementwise product `C(x) = A(x) B(x)` of two tensor trains with the same site dimensions, by
the swap network of arXiv:2410.19747, Fig. 1B. `A` and `B` are vectors of three-leg cores laid
out `(left, site, right)` with dimension-one boundary bonds, or
`TensorCrossInterpolation.TensorTrain{T,3}`; the result has the kind of the inputs and the
promoted element type. Inputs are not modified.

**`tolerance` mode**, the one to use: the relative L2 error of the returned product is
measured, not assumed. A pilot pass runs at `pilot_cutoff`, whose default assumes the error
scales like `(N - 1) sqrt(cutoff)` (the number of bonds; measured on sharp products); the
exact residual `C(x) - A(x) B(x)` is sampled at `nsamples` random points; if
`estimate + 2 stderr > tolerance`, the ratio `kappa = estimate / sqrt(cutoff)`, which is
nearly cutoff-independent for a given pair of trains, sets the next cutoff
`safety * (tolerance / kappa)^2`, up to `max_attempts` passes. Tightening stops early once the
bond cap binds or the estimate stops improving by more than its own standard error, since
neither can be helped by a smaller cutoff. The result carries `info.error_estimate`,
`info.error_stderr` and `info.verified`; an unverified product (a bond cap, or the attempt
limit) is returned, not thrown, and the caller decides.

Verification also needs the sample to have carried the error: `info.ess_num` must reach
`min_ess`, unless `info.error_bound` meets the tolerance on its own and no sampling is needed.
That refuses a product whose residual the sample saw once or not at all. It is a floor, not a
proof: a residual on a vanishing fraction of the grid sitting above a uniform roundoff floor
keeps `ess_num` high and is certified, so `verified` means the sampled residual is below the
tolerance at this sample size, not that the true error is.

**`cutoff` mode**, for expert use: one pass at a fixed relative discarded squared weight per
SVD (the paper's `epsilon`, ITensors' relative `cutoff`). What it does *not* promise: the
total error of a many-truncation algorithm is a problem-dependent multiple of
`sqrt(cutoff)`, about the number of bonds on sharp products. Pass `nsamples > 0` to measure it.

Exactly one of `tolerance` and `cutoff` must be given. `maxbonddim` caps every bond the
algorithm creates, inside the truncation. With `final_truncation = true` the result receives
one more truncating sweep and is returned left-canonical with the norm on its last core.
The measurement covers this multiplication's own truncations; errors the inputs already carry
are inherited by any product. Cost `O(N^2 d^3 chi^3)`, memory `O(d^2 chi^2)` with
`chi = info.peak_bonddim`; tightening the tolerance grows `chi` steeply, so read
`info.cutoffs[end]` and `info.peak_bonddim` when a product is slow.
"""
function multiply(A::AbstractVector{<:AbstractArray{TA,3}}, B::AbstractVector{<:AbstractArray{TB,3}};
                  cutoff::Union{Nothing,Real} = nothing, tolerance::Union{Nothing,Real} = nothing,
                  maxbonddim::Integer = typemax(Int), final_truncation::Bool = true,
                  nsamples::Union{Nothing,Integer} = nothing, rng::AbstractRNG = Random.default_rng(),
                  pilot_cutoff::Union{Nothing,Real} = nothing, safety::Real = 0.5,
                  max_attempts::Integer = 3, min_ess::Integer = 30) where {TA<:Number,TB<:Number}
    (cutoff === nothing) == (tolerance === nothing) &&
        throw(ArgumentError("give exactly one of `cutoff` and `tolerance`; got cutoff = $(cutoff), tolerance = $(tolerance)"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be at least 1, got $(maxbonddim)"))
    min_ess >= 0 || throw(ArgumentError("min_ess must be non-negative, got $(min_ess)"))
    _check_inputs(A, B)
    N = length(A)
    if cutoff !== nothing
        cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative, got $(cutoff)"))
        ns = nsamples === nothing ? 0 : Int(nsamples)
        ns == 0 || ns >= 100 || throw(ArgumentError("nsamples must be 0 or at least 100, got $(ns)"))
        cores, pass = _multiply_pass(A, B, Float64(cutoff), maxbonddim, final_truncation)
        est, se, en, ed = ns == 0 ? (NaN, NaN, NaN, NaN) : sampled_relative_error(A, B, cores, rng, ns)
        return cores, MultiplyInfo(pass.peak_bonddim, pass.peak_bonddim_per_step, pass.bonddims,
                                   pass.n_swaps, pass.discarded_weight, pass.hit_maxbonddim,
                                   pass.error_bound, est, se, false, 1, [Float64(cutoff)], ns,
                                   en, ed, Int(min_ess))
    end
    tolerance > 0 || throw(ArgumentError("tolerance must be positive, got $(tolerance)"))
    0 < safety <= 1 || throw(ArgumentError("safety must lie in (0, 1], got $(safety)"))
    max_attempts >= 1 || throw(ArgumentError("max_attempts must be at least 1, got $(max_attempts)"))
    ns = nsamples === nothing ? 4000 : Int(nsamples)
    ns >= 100 || throw(ArgumentError("nsamples must be at least 100 in tolerance mode, got $(ns)"))
    pilot = pilot_cutoff === nothing ? Float64(tolerance)^2 / max(N - 1, 1)^2 : Float64(pilot_cutoff)
    pilot > 0 || throw(ArgumentError("pilot_cutoff must be positive, got $(pilot)"))
    cutoffs = [pilot]
    cores, pass = _multiply_pass(A, B, pilot, maxbonddim, final_truncation)
    est, se, en, ed = sampled_relative_error(A, B, cores, rng, ns)
    while est + 2se > tolerance && length(cutoffs) < max_attempts
        # A capped bond, and an estimate that has stopped moving, are both insensitive to the
        # cutoff: another pass costs a full O(N^2 d^3 chi^3) and cannot lower the error.
        pass.hit_maxbonddim && break
        kappa = est / sqrt(cutoffs[end])
        next = isfinite(kappa) ? safety * (Float64(tolerance) / kappa)^2 : cutoffs[end] / 100
        push!(cutoffs, max(next, eps(Float64)))
        previous = est
        cores, pass = _multiply_pass(A, B, cutoffs[end], maxbonddim, final_truncation)
        est, se, en, ed = sampled_relative_error(A, B, cores, rng, ns)
        previous - est > se || break
    end
    informative = en >= min_ess || pass.error_bound <= tolerance
    return cores, MultiplyInfo(pass.peak_bonddim, pass.peak_bonddim_per_step, pass.bonddims,
                               pass.n_swaps, pass.discarded_weight, pass.hit_maxbonddim,
                               pass.error_bound, est, se, est + 2se <= tolerance && informative,
                               length(cutoffs), cutoffs, ns, en, ed, Int(min_ess))
end

function multiply(A::TCI.TensorTrain{TA,3}, B::TCI.TensorTrain{TB,3}; kwargs...) where {TA,TB}
    cores, info = multiply(A.sitetensors, B.sitetensors; kwargs...)
    return TCI.TensorTrain{promote_type(TA, TB),3}(cores), info
end
