@testset "left_canonical! and sweep_truncate!" begin
    LC = TensorTrainMultiplication.left_canonical!
    ST = TensorTrainMultiplication.sweep_truncate!
    rng = MersenneTwister(11)
    sitedims = [2, 3, 2, 2, 3]
    for T in (Float64, ComplexF64)
        A = random_train(rng, T, sitedims, 4)
        v = dense_vector(A)

        cores = LC([copy(a) for a in A])
        @test dense_vector(cores) ≈ v
        for p in 1:(length(cores) - 1)
            l, d, r = size(cores[p])
            Q = reshape(cores[p], l * d, r)
            @test Q' * Q ≈ I(r) atol = 1e-12
        end
        @test all(size(cores[p], 3) <= size(A[p], 3) for p in 1:(length(A) - 1))

        # The swap network leaves the centre on the first core and the rest right-orthogonal;
        # reversing a left-canonical train produces exactly that.
        rev = [permutedims(c, (3, 2, 1)) for c in reverse(LC([copy(a) for a in A]))]
        vrev = dense_vector(rev)
        w, capped, sqrtsum = ST(rev, 0.0, typemax(Int))
        @test sqrtsum == 0.0
        @test dense_vector(rev) ≈ vrev
        @test w == 0.0 && !capped
        bonds = [size(c, 3) for c in rev[1:(end - 1)]]
        @test all(bonds .<= rank_bounds(reverse(sitedims)))
        @test all(bonds .<= 4)
        for p in 1:(length(rev) - 1)
            l, d, r = size(rev[p])
            Q = reshape(rev[p], l * d, r)
            @test Q' * Q ≈ I(r) atol = 1e-12
        end

        w, capped, sqrtsum = ST(rev, 0.0, 2)
        @test sqrtsum >= sqrt(w)
        @test capped && all(size(c, 3) <= 2 for c in rev[1:(end - 1)])
        @test w > 0
    end
end
