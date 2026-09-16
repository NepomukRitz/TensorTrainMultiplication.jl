# TensorTrainMultiplication.jl

The elementwise (Hadamard) product of two tensor trains through the swap network of
Michailidis, Fenton and Kiffner, *Tensor Train Multiplication*, arXiv:2410.19747.

```julia
using TensorTrainMultiplication
C, info = multiply(A, B; cutoff = 1e-9)
```

`A` and `B` are `Vector{<:AbstractArray{T,3}}` cores laid out `(left, site, right)` with
dimension-one boundary bonds, or `TensorCrossInterpolation.TensorTrain{T,3}`. The result has
the kind of the inputs. Element types are generic and promote.

## Why

The conventional way to multiply two trains promotes one of them to a diagonal operator and
applies it to the other with a variational fit: cost `O(N d chi^4)`, memory `O(chi^3)`
(the fit's environments). TTM never forms the operator. It concatenates the two trains into
one chain, swaps corresponding carriages next to each other with one truncated SVD per swap,
and contracts each pair through a COPY tensor as soon as it is adjacent: cost
`O(N^2 d^3 chi^3)`, memory `O(d^2 chi^2)`, in the largest *intermediate* bond dimension.
That intermediate bond grows above the input rank during the swaps (`info.peak_bonddim`
reports it) and decides whether TTM wins; on quantics Green's functions at a relative
discarded weight of `1e-9` it measured 3.3-3.5 times the input rank, and TTM was 2.1x
faster at input rank 112 and 4.5x at 223 with no increase in peak memory where the fit
added 9 GB (ReFrequenTT, `docs/notes/analyses/ttm_product_benchmark.md`).

## Accuracy semantics

`cutoff` is the relative discarded squared weight per truncated SVD,

    sum_{dropped} s_i^2 / sum_all s_i^2 <= cutoff,

which is `epsilon` of the paper's Eq. 10 and ITensors' relative `cutoff`. The paper finds a
squared relative error of about `10 cutoff` for the whole product, i.e. a relative L2 error
of about `sqrt(10 cutoff)`. The package tests assert that with a factor 3 of allowance on a
smooth product and a factor 10 on a Lorentzian times a tanh step; products with sharp
features sit nearer the upper end. `info.discarded_weight` is the sum of the relative
discarded weights of every truncation; because every truncation is a global one (the
swapped pair holds the orthogonality centre), the triangle inequality bounds the squared
relative error by `n_swaps` times it. `maxbonddim` caps every bond the algorithm creates,
intermediate ones included, inside the truncation; what it removes is reported in
`info.discarded_weight` and `info.hit_maxbonddim`, never thrown. `final_truncation = true`
(default) adds one truncating sweep and returns the result left-canonical.

## API

- `multiply(A, B; cutoff = 0.0, maxbonddim = typemax(Int), final_truncation = true) -> (C, info)`
- `MultiplyInfo`: `peak_bonddim`, `peak_bonddim_per_step`, `bonddims`, `n_swaps`,
  `discarded_weight`, `hit_maxbonddim`.

Threading is BLAS threading; the swap SVDs dominate and are LAPACK-bound.

## Related

`AlternatingCrossInterpolation.jl` interpolates the product instead of forming it; it wins
where the product is much simpler than its factors and loses where it is not. `BubbleTeaCI`
exposes this package as `contract(...; alg = "ttm")`.
