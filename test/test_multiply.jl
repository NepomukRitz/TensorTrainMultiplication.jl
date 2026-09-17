@testset "multiply reproduces the dense elementwise product" begin
    rng = MersenneTwister(3)
    for T in (Float64, ComplexF64), sitedims in ([2, 2, 2, 2, 2, 2], [2, 3, 1, 4, 2, 3], [3, 3], [5])
        A = random_train(rng, T, sitedims, 3)
        B = random_train(rng, T, sitedims, 4)
        truth = dense_vector(A) .* dense_vector(B)
        C, info = multiply(A, B; cutoff = 0.0)
        N = length(sitedims)
        @test length(C) == N
        @test [size(c, 2) for c in C] == sitedims
        @test size(C[1], 1) == 1 && size(C[end], 3) == 1
        @test eltype(C[1]) == T
        @test dense_vector(C) ≈ truth atol = 1e-10 * norm(truth)
        # `cutoff` defaults to 0.0, the exact product
        @test dense_vector(first(multiply(A, B))) == dense_vector(C)
        @test info.n_swaps == N * (N - 1) ÷ 2
        @test length(info.peak_bonddim_per_step) == N
        @test info.peak_bonddim == maximum(info.peak_bonddim_per_step)
        @test info.bonddims == [size(c, 3) for c in C[1:(end - 1)]]
        @test !info.hit_maxbonddim
        # Only the noise-floor rule can discard anything at cutoff 0: weights of order eps^2.
        @test info.discarded_weight < 1e-24
        # Every bond of the result is an exact rank of a bipartition of the product.
        @test all(info.bonddims .<= rank_bounds(sitedims))
    end
end

@testset "mixed element types promote" begin
    rng = MersenneTwister(5)
    A = random_train(rng, Float64, [2, 2, 2, 2], 2)
    B = random_train(rng, ComplexF64, [2, 2, 2, 2], 3)
    C, _ = multiply(A, B; cutoff = 0.0)
    @test eltype(C[1]) == ComplexF64
    @test dense_vector(C) ≈ dense_vector(A) .* dense_vector(B)
end

@testset "TensorTrain method" begin
    rng = MersenneTwister(7)
    A = TCI.TensorTrain(random_train(rng, Float64, [2, 2, 2, 2, 2], 3))
    B = TCI.TensorTrain(random_train(rng, Float64, [2, 2, 2, 2, 2], 2))
    C, info = multiply(A, B; cutoff = 0.0)
    @test C isa TCI.TensorTrain{Float64,3}
    @test dense_vector(C.sitetensors) ≈ dense_vector(A.sitetensors) .* dense_vector(B.sitetensors)
    @test TCI.linkdims(C) == info.bonddims
end

@testset "final_truncation = false keeps the values at larger bonds" begin
    rng = MersenneTwister(9)
    A = random_train(rng, Float64, fill(2, 7), 3)
    B = random_train(rng, Float64, fill(2, 7), 3)
    C1, i1 = multiply(A, B; cutoff = 0.0, final_truncation = true)
    C0, i0 = multiply(A, B; cutoff = 0.0, final_truncation = false)
    @test dense_vector(C0) ≈ dense_vector(C1)
    @test all(i0.bonddims .>= i1.bonddims)
    # The swaps leave bonds above the exact ranks of the product; only the final sweep brings
    # them down. Without this, a `final_truncation` that is never read would pass the testset.
    @test any(i0.bonddims .> i1.bonddims)
end

@testset "multiply leaves its inputs untouched" begin
    rng = MersenneTwister(11)
    A = random_train(rng, Float64, fill(2, 5), 3)
    B = random_train(rng, Float64, fill(2, 5), 2)
    A0 = deepcopy(A)
    B0 = deepcopy(B)
    multiply(A, B; cutoff = 1e-8)
    @test A == A0
    @test B == B0
end

@testset "truncation on smooth quantics functions" begin
    R = 10
    A = quantics_train(x -> exp(-3x), R)
    B = quantics_train(x -> 1 / (1 + x), R)
    truth = quantics_values(x -> exp(-3x) / (1 + x), R)
    errors = Float64[]
    weights = Float64[]
    for cutoff in (1e-4, 1e-8, 1e-12)
        C, info = multiply(A, B; cutoff = cutoff)
        push!(errors, norm(dense_vector(C) - truth) / norm(truth))
        push!(weights, info.discarded_weight)
        # arXiv:2410.19747 finds Gamma ~ 10 epsilon for the squared relative error, i.e. a
        # relative error of about sqrt(10 cutoff); a factor 3 on top is the allowance.
        @test errors[end] <= 3 * sqrt(10 * cutoff)
    end
    @test issorted(errors; rev = true)
    @test issorted(weights; rev = true)
    @test errors[end] < 1e-5
end

@testset "gauge sweep keeps the transient near the input rank" begin
    R = 12
    for (f, g) in ((x -> exp(-200(x - 0.4)^2), x -> exp(-200(x - 0.41)^2)),
                   (x -> 1 / ((x - 0.5)^2 + 1e-3), x -> tanh((x - 0.5) / 0.05)))
        A = quantics_train(f, R)
        B = quantics_train(g, R)
        truth = quantics_values(x -> f(x) * g(x), R)
        C, info = multiply(A, B; cutoff = 1e-8)
        rank_in = max(maximum(size.(A, 3)), maximum(size.(B, 3)))
        # Without the QR gauge sweep every swap SVD truncates against a non-orthonormal
        # environment and keeps 4-5x more singular values than the product needs, at the same
        # accuracy; with the sweep the peak stays at the input rank on these products. This is
        # the assertion that notices a missing sweep; the cutoff-0 tests cannot.
        @test info.peak_bonddim <= 2 * rank_in
        @test norm(dense_vector(C) - truth) / norm(truth) <= 10 * sqrt(10 * 1e-8)
    end
end

@testset "maxbonddim caps every bond and reports it" begin
    rng = MersenneTwister(13)
    A = random_train(rng, Float64, fill(2, 8), 4)
    B = random_train(rng, Float64, fill(2, 8), 4)
    Cfull, ifull = multiply(A, B; cutoff = 0.0)
    @test ifull.peak_bonddim > 3
    C, info = multiply(A, B; cutoff = 0.0, maxbonddim = 3)
    @test info.hit_maxbonddim
    # peak_bonddim counts the inputs' own bonds (4 here) while they are still in the chain;
    # every bond the algorithm creates is capped at 3.
    @test info.peak_bonddim <= max(3, 4)
    @test all(info.bonddims .<= 3)
    @test all(size(c, 3) <= 3 for c in C)
    @test info.discarded_weight > 0
    @test norm(dense_vector(C) - dense_vector(Cfull)) > 0
    @test length(dense_vector(C)) == 2^8
end

@testset "argument errors name the problem" begin
    rng = MersenneTwister(17)
    A = random_train(rng, Float64, [2, 2, 2], 2)
    B = random_train(rng, Float64, [2, 3, 2], 2)
    @test_throws DimensionMismatch multiply(A, B; cutoff = 0.0)
    @test_throws DimensionMismatch multiply(A, A[1:2]; cutoff = 0.0)
    @test_throws ArgumentError multiply(A, A; cutoff = -1.0)
    @test_throws ArgumentError multiply(A, A; cutoff = 0.0, maxbonddim = 0)
    bad = [copy(a) for a in A]
    bad[1] = randn(rng, 2, 2, size(A[1], 3))                 # left boundary bond of dimension 2
    @test_throws ArgumentError multiply(bad, A; cutoff = 0.0)
    broken = [copy(a) for a in A]
    broken[2] = randn(rng, size(A[2], 1) + 1, 2, size(A[2], 3))   # inconsistent internal bond
    @test_throws DimensionMismatch multiply(broken, A; cutoff = 0.0)
    @test_throws ArgumentError multiply(Array{Float64,3}[], Array{Float64,3}[]; cutoff = 0.0)
end

@testset "error_bound is a rigorous bound on the relative L2 error" begin
    rng = MersenneTwister(21)
    A = random_train(rng, Float64, fill(2, 8), 4)
    B = random_train(rng, Float64, fill(2, 8), 4)
    truth = dense_vector(A) .* dense_vector(B)
    for cutoff in (1e-2, 1e-4, 1e-6)
        C, info = multiply(A, B; cutoff = cutoff)
        err = norm(dense_vector(C) - truth) / norm(truth)
        @test err <= info.error_bound
        @test info.error_bound <= (info.n_swaps + 7) * sqrt(cutoff) + 1e-10
    end
    R = 12
    for (f, g) in ((x -> 1 / ((x - 0.5)^2 + 1e-3), x -> tanh((x - 0.5) / 0.05)),
                   (x -> (x - 0.5) / ((x - 0.5)^2 + 0.02^2), x -> tanh((x - 0.5) / 0.02)))
        A = quantics_train(f, R)
        B = quantics_train(g, R)
        truth = quantics_values(x -> f(x) * g(x), R)
        for cutoff in (1e-6, 1e-10)
            C, info = multiply(A, B; cutoff = cutoff)
            err = norm(dense_vector(C) - truth) / norm(truth)
            @test err <= info.error_bound <= 30 * err
        end
    end
end
