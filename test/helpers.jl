# test/helpers.jl

"""Full tensor of a train as a vector, first site fastest."""
function dense_vector(cores)
    M = reshape(cores[1], size(cores[1], 2), size(cores[1], 3))
    for c in cores[2:end]
        l, d, r = size(c)
        M = reshape(reshape(M, :, l) * reshape(c, l, d * r), :, r)
    end
    return vec(M)
end

function random_train(rng, ::Type{T}, sitedims, bond) where {T}
    N = length(sitedims)
    bonds = [1; fill(bond, N - 1); 1]
    return [randn(rng, T, bonds[i], sitedims[i], bonds[i + 1]) for i in 1:N]
end

"""Largest possible rank of each bipartition of a train with these site dimensions."""
rank_bounds(sitedims) = [min(prod(sitedims[1:k]), prod(sitedims[(k + 1):end])) for k in 1:(length(sitedims) - 1)]

"""Quantics train of `f` on [0, 1) with `R` bits, site 1 the most significant bit."""
function quantics_train(f, R; tolerance = 1e-14)
    x(bits) = sum((bits[j] - 1) * 2.0^(-j) for j in 1:R)
    tci, _, _ = TCI.crossinterpolate2(Float64, bits -> f(x(bits)), fill(2, R); tolerance = tolerance)
    return TCI.TensorTrain(tci).sitetensors
end

"""Values of `f` on the same 2^R points, in `dense_vector` order."""
quantics_values(f, R) = [f(sum(((n >> (j - 1)) & 1) * 2.0^(-j) for j in 1:R)) for n in 0:(2^R - 1)]
