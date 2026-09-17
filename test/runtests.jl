# test/runtests.jl
using Test
using Aqua
using LinearAlgebra
using Random
using TensorTrainMultiplication
import TensorCrossInterpolation as TCI

include("helpers.jl")

@testset "TensorTrainMultiplication" begin
    include("test_truncation.jl")
    include("test_canonical.jl")
    include("test_multiply.jl")
    include("test_aqua.jl")
end
