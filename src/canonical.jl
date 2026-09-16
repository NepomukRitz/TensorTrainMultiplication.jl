"""
    left_canonical!(cores) -> cores

QR sweep from the first core to the last. Afterwards every core but the last is
left-orthogonal (its left bond and site index contracted with its own conjugate give the
identity on its right bond) and the last core carries the norm. Bond dimensions can only
shrink. Overwrites the entries of `cores` with new arrays; the arrays it received are not
kept, so pass copies of anything you still need.
"""
function left_canonical!(cores::Vector{Array{T,3}}) where {T}
    for p in 1:(length(cores) - 1)
        l, d, r = size(cores[p])
        F = qr!(reshape(cores[p], l * d, r))
        Q = Matrix(F.Q)
        k = size(Q, 2)
        cores[p] = reshape(Q, l, d, k)
        l2, d2, r2 = size(cores[p + 1])
        cores[p + 1] = reshape(F.R * reshape(cores[p + 1], l2, d2 * r2), k, d2, r2)
    end
    return cores
end

"""
    sweep_truncate!(cores, cutoff, maxbonddim) -> (discarded, capped)

One SVD sweep from the first core to the last for a train whose orthogonality centre is on
the first core and whose other cores are right-orthogonal. Every bond is truncated with
[`truncated_rank`](@ref); the train ends left-canonical with the norm on the last core.
Returns the summed relative discarded weight and whether `maxbonddim` removed anything.
"""
function sweep_truncate!(cores::Vector{Array{T,3}}, cutoff::Real, maxbonddim::Integer) where {T}
    discarded = 0.0
    capped = false
    for p in 1:(length(cores) - 1)
        l, d, r = size(cores[p])
        F = svd!(reshape(cores[p], l * d, r))
        k, w, c = truncated_rank(F.S, cutoff, maxbonddim)
        discarded += w
        capped |= c
        cores[p] = reshape(F.U[:, 1:k], l, d, k)
        carry = Diagonal(F.S[1:k]) * F.Vt[1:k, :]
        l2, d2, r2 = size(cores[p + 1])
        cores[p + 1] = reshape(carry * reshape(cores[p + 1], l2, d2 * r2), k, d2, r2)
    end
    return discarded, capped
end
