"""
    evaluate_cores(cores, idx) -> value

The train's entry at the multi-index `idx` (one site index per core), by contracting the cores
left to right: `O(N chi^2)`.
"""
function evaluate_cores(cores::AbstractVector{<:AbstractArray{T,3}}, idx::AbstractVector{<:Integer}) where {T}
    v = Vector{T}(vec(@view cores[1][1, idx[1], :]))
    for k in 2:length(cores)
        v = vec(transpose(v) * @view cores[k][:, idx[k], :])
    end
    return only(v)
end

"""
    sampled_relative_error(A, B, C, rng, nsamples) -> (estimate, stderr)

Sampled estimate of the relative L2 error `||C - A o B||_2 / ||A o B||_2` over the
full index grid, from `nsamples` uniform random multi-indices, with its standard error by
the delta method on the numerator. `A(x) B(x)` is exact for the trains as given, so the
estimate needs no reference computation. Returns `(0.0, 0.0)` when both `C` and `A o B`
vanish on every sample and `(Inf, 0.0)` when `A o B` does but `C` does not.
"""
function sampled_relative_error(A, B, C, rng::AbstractRNG, nsamples::Integer)
    N = length(C)
    sitedims = [size(c, 2) for c in C]
    idx = Vector{Int}(undef, N)
    num = zeros(Float64, nsamples)
    den = zeros(Float64, nsamples)
    for m in 1:nsamples
        for k in 1:N
            idx[k] = rand(rng, 1:sitedims[k])
        end
        ab = evaluate_cores(A, idx) * evaluate_cores(B, idx)
        num[m] = abs2(evaluate_cores(C, idx) - ab)
        den[m] = abs2(ab)
    end
    X = sum(num) / nsamples
    Y = sum(den) / nsamples
    Y == 0 && return (X == 0 ? 0.0 : Inf, 0.0)
    est = sqrt(X / Y)
    X == 0 && return (0.0, 0.0)
    sx = sqrt(sum(abs2, num .- X) / (nsamples - 1) / nsamples)
    return est, sx / (2 * sqrt(X * Y))
end
