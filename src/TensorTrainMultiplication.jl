"""
Tensor Train Multiplication: the elementwise (Hadamard) product of two tensor trains through
the swap network of Michailidis, Fenton and Kiffner, arXiv:2410.19747.

The algorithm never promotes a factor to an operator. It concatenates the two trains into one
chain, swaps corresponding carriages next to each other with one truncated SVD per swap, and
contracts each adjacent pair through a COPY tensor as soon as it is adjacent. Cost
`O(N^2 d^3 chi^3)` and memory `O(d^2 chi^2)` in the largest intermediate bond dimension
`chi`, against `O(N d chi^4)` and `O(chi^3)` for the conventional MPO fit.

See [`multiply`](@ref).
"""
module TensorTrainMultiplication

using LinearAlgebra
using Random
import TensorCrossInterpolation as TCI

export multiply, MultiplyInfo

include("truncation.jl")
include("canonical.jl")
include("sampling.jl")
include("multiply.jl")

end
