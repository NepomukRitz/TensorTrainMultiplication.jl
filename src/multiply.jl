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
"""
struct MultiplyInfo
    peak_bonddim::Int
    peak_bonddim_per_step::Vector{Int}
    bonddims::Vector{Int}
    n_swaps::Int
    discarded_weight::Float64
    hit_maxbonddim::Bool
end

function _check_inputs(A, B, cutoff, maxbonddim)
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
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative, got $(cutoff)"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be at least 1, got $(maxbonddim)"))
    return N
end

"""
    multiply(A, B; cutoff = 0.0, maxbonddim = typemax(Int), final_truncation = true)
        -> (C, info::MultiplyInfo)

Elementwise product `C(x) = A(x) B(x)` of two tensor trains with the same site dimensions,
by the swap network of arXiv:2410.19747, Fig. 1B.

`A` and `B` are vectors of three-leg cores laid out `(left, site, right)` with dimension-one
boundary bonds, or `TensorCrossInterpolation.TensorTrain{T,3}`; the result has the kind of
the inputs and the promoted element type. Inputs are not modified.

`cutoff` is the relative discarded squared weight allowed at every truncated SVD: the paper's
`epsilon` (Eq. 10) and ITensors' relative `cutoff`, so a value tuned for one carries over
to the other. The paper measures a squared relative error of about `10 cutoff`, i.e. a
relative L2 error of about `sqrt(10 cutoff)`. `maxbonddim` caps every bond, including the
intermediate ones, inside the SVD truncation, so it bounds cost and memory; what it removes is
reported in `info.discarded_weight` and `info.hit_maxbonddim` rather than thrown. With
`final_truncation = true` the result receives one more truncating sweep and is returned
left-canonical with the norm on its last core.

Cost `O(N^2 d^3 chi^3)`, memory `O(d^2 chi^2)`, with `chi` the largest intermediate bond
dimension, `info.peak_bonddim`.
"""
function multiply(A::AbstractVector{<:AbstractArray{TA,3}}, B::AbstractVector{<:AbstractArray{TB,3}};
                  cutoff::Real = 0.0, maxbonddim::Integer = typemax(Int),
                  final_truncation::Bool = true) where {TA<:Number,TB<:Number}
    T = promote_type(TA, TB)
    N = _check_inputs(A, B, cutoff, maxbonddim)
    if N == 1
        d = size(A[1], 2)
        C = Array{T,3}(undef, 1, d, 1)
        for s in 1:d
            C[1, s, 1] = A[1][1, s, 1] * B[1][1, s, 1]
        end
        return [C], MultiplyInfo(1, [1], Int[], 0, 0.0, false)
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
        w, c = sweep_truncate!(cores, cutoff, maxbonddim)
        discarded += w
        capped |= c
    end
    info = MultiplyInfo(maximum(peak_per_step), peak_per_step,
                        [size(c, 3) for c in cores[1:(N - 1)]], N * (N - 1) ÷ 2, discarded, capped)
    return cores, info
end

function multiply(A::TCI.TensorTrain{TA,3}, B::TCI.TensorTrain{TB,3}; kwargs...) where {TA,TB}
    cores, info = multiply(A.sitetensors, B.sitetensors; kwargs...)
    return TCI.TensorTrain{promote_type(TA, TB),3}(cores), info
end
