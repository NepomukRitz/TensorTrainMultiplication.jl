@testset "evaluate_cores reads the train at one multi-index" begin
    rng = MersenneTwister(31)
    for T in (Float64, ComplexF64)
        sitedims = [2, 3, 2, 4, 2]
        A = random_train(rng, T, sitedims, 3)
        v = dense_vector(A)
        for _ in 1:20
            idx = [rand(rng, 1:d) for d in sitedims]
            lin = 1 + sum((idx[k] - 1) * prod(sitedims[1:(k - 1)]) for k in eachindex(idx))
            @test TensorTrainMultiplication.evaluate_cores(A, idx) ≈ v[lin]
        end
    end
end

@testset "sampled_relative_error is unbiased with a usable standard error" begin
    rng = MersenneTwister(37)
    R = 12
    for (f, g) in ((x -> exp(-3x), x -> 1 / (1 + x)),
                   (x -> 1 / ((x - 0.5)^2 + 1e-3), x -> tanh((x - 0.5) / 0.05)),
                   (x -> (x - 0.5) / ((x - 0.5)^2 + 0.02^2), x -> tanh((x - 0.5) / 0.02)))
        A = quantics_train(f, R)
        B = quantics_train(g, R)
        truth = quantics_values(x -> f(x) * g(x), R)
        C, _ = multiply(A, B; cutoff = 1e-8)
        err = norm(dense_vector(C) - truth) / norm(truth)
        est, se = TensorTrainMultiplication.sampled_relative_error(A, B, C, rng, 8000)
        @test abs(est - err) <= 3 * se + 1e-12
        @test 0 < se < 0.3 * est
    end
    # an untruncated product of rank-one trains carries only roundoff, and the estimate sees it
    A = [reshape([1.0, exp(-2.0^-j)], 1, 2, 1) for j in 1:10]
    B = [reshape([1.0, exp(-2 * 2.0^-j)], 1, 2, 1) for j in 1:10]
    C, _ = multiply(A, B; cutoff = 0.0)
    est, se = TensorTrainMultiplication.sampled_relative_error(A, B, C, rng, 500)
    @test est < 1e-13 && se < 1e-13
    # the X == 0 branch: a bitwise-exact C gives exactly (0.0, 0.0)
    @test TensorTrainMultiplication.sampled_relative_error(C, [reshape([1.0, 1.0], 1, 2, 1) for _ in 1:10], C, rng, 200) == (0.0, 0.0)
end
