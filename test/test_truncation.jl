@testset "truncated_rank" begin
    TR = TensorTrainMultiplication.truncated_rank
    s = [1.0, 0.5, 0.1, 1e-3, 1e-8]
    total = sum(abs2, s)

    k, w, capped = TR(s, 0.0, typemax(Int))            # cutoff 0 keeps everything real
    @test (k, w, capped) == (5, 0.0, false)

    k, w, capped = TR(s, 1e-5, typemax(Int))           # tail 1e-16 + 1e-6 fits, 0.01 does not
    @test k == 3 && !capped
    @test w ≈ (1e-6 + 1e-16) / total

    k, w, capped = TR(s, 1e-5, 2)                      # the cap removes on top and reports it
    @test k == 2 && capped
    @test w ≈ (1e-6 + 1e-16 + 0.01) / total

    k, w, capped = TR(s, 1.0, typemax(Int))            # at least one value always survives
    @test k == 1 && w ≈ 1 - 1 / total

    k, _, _ = TR([1.0, 1e-17, 0.0], 0.0, typemax(Int)) # noise and exact zeros go at cutoff 0
    @test k == 1

    @test TR(Float64[], 0.0, 3) == (0, 0.0, false)
    @test TR([0.0, 0.0], 0.0, 3) == (1, 0.0, false)
end
