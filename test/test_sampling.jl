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

@testset "sampled_relative_error is consistent with a usable standard error" begin
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
        est, se, ess_num, ess_den = TensorTrainMultiplication.sampled_relative_error(A, B, C, rng, 8000)
        @test abs(est - err) <= 3 * se + 1e-12
        @test 0 < se < 0.3 * est
        @test 30 <= ess_num <= 8000 && 30 <= ess_den <= 8000
    end
    # an untruncated product of rank-one trains carries only roundoff, and the estimate sees it
    A = [reshape([1.0, exp(-2.0^-j)], 1, 2, 1) for j in 1:10]
    B = [reshape([1.0, exp(-2 * 2.0^-j)], 1, 2, 1) for j in 1:10]
    C, _ = multiply(A, B; cutoff = 0.0)
    est, se, _, _ = TensorTrainMultiplication.sampled_relative_error(A, B, C, rng, 500)
    @test est < 1e-13 && se < 1e-13
    # the X == 0 branch: a bitwise-exact C gives exactly (0.0, 0.0) with no residual mass
    exact = TensorTrainMultiplication.sampled_relative_error(
        C, [reshape([1.0, 1.0], 1, 2, 1) for _ in 1:10], C, rng, 200)
    @test exact[1:3] == (0.0, 0.0, 0.0) && exact[4] > 30
end

@testset "effective sample size reports how much of the sample carries the mass" begin
    @test TensorTrainMultiplication.effective_sample_size(zeros(100)) == 0.0
    @test TensorTrainMultiplication.effective_sample_size(fill(3.0, 100)) ≈ 100
    @test TensorTrainMultiplication.effective_sample_size([1e6; zeros(999)]) ≈ 1
    # a residual living on a vanishing fraction of the grid: 20 sites, A = 1 + 10 [first 13
    # site indices are 1], B = 1, C the same with the spike dropped. The true relative L2
    # error is 0.1097, and a uniform sample of 4000 points reports exactly 0 most of the time.
    N, k = 20, 13
    A = Vector{Array{Float64,3}}(undef, N)
    c = zeros(1, 2, 2); c[1, :, 1] .= 1.0; c[1, 1, 2] = 1.0; A[1] = c
    for j in 2:(N - 1)
        c = zeros(2, 2, 2); c[1, :, 1] .= 1.0
        j <= k ? (c[2, 1, 2] = 1.0) : (c[2, :, 2] .= 1.0)
        A[j] = c
    end
    c = zeros(2, 2, 1); c[1, :, 1] .= 1.0; c[2, :, 1] .= 10.0; A[N] = c
    ones_train = [reshape([1.0, 1.0], 1, 2, 1) for _ in 1:N]
    spike = 2.0^(N - k)
    @test sqrt(spike * 100 / ((2.0^N - spike) + spike * 121)) ≈ 0.1096849982679642
    blind = count(1:20) do seed
        est, se, ess_num, _ = TensorTrainMultiplication.sampled_relative_error(
            A, ones_train, ones_train, MersenneTwister(seed), 4000)
        est == 0.0 && se == 0.0 && ess_num == 0.0
    end
    @test blind > 10
end
