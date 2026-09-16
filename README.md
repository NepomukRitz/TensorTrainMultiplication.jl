# TensorTrainMultiplication.jl

The elementwise (Hadamard) product of two tensor trains through the swap network of
Michailidis, Fenton and Kiffner, *Tensor Train Multiplication*, arXiv:2410.19747.

```julia
using TensorTrainMultiplication
C, info = multiply(A, B; tolerance = 1e-6)    # the error is measured, not assumed
info.verified, info.error_estimate, info.peak_bonddim
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

Exactly one of `tolerance` and `cutoff` must be given; there is no default.

**`tolerance`** is a relative L2 error of the whole product, `||C - A o B||_2 / ||A o B||_2`,
and it is *measured*. The exact residual `C(x) - A(x) B(x)` is sampled at `nsamples` random
multi-indices -- no reference product is ever formed, because `A(x) B(x)` is exact for the
trains as given -- and the cutoff is tightened from a pilot pass until
`error_estimate + 2 error_stderr <= tolerance`. The result carries `info.error_estimate`,
`info.error_stderr` and `info.verified`; a product that a bond cap or the attempt limit left
unverified is returned, not thrown, and the caller decides.

**`cutoff`** is for expert use: the relative discarded squared weight per truncated SVD,

    sum_{dropped} s_i^2 / sum_all s_i^2 <= cutoff,

which is `epsilon` of the paper's Eq. 10 and ITensors' relative `cutoff`. It bounds *one*
truncation. A train has many bonds and the errors add, so the total is a problem-dependent
multiple of `sqrt(cutoff)` -- on sharp quantics products, about the number of bonds. Pass
`nsamples > 0` to measure it.

`info.error_bound` is the rigorous `sum_i sqrt(w_i)` over every truncation: each happens in
mixed-canonical gauge, so its error is exactly `sqrt(w_i)` times the current norm and the
triangle inequality sums them. It is always valid and pessimistic; a diagnostic, not the
control. `info.discarded_weight` is the sum of the relative discarded weights themselves.
`maxbonddim` caps every bond the algorithm creates, intermediate ones included, inside the
truncation; what it removes is reported in `info.discarded_weight` and `info.hit_maxbonddim`,
never thrown. `final_truncation = true` (default) adds one truncating sweep and returns the
result left-canonical.

## API

- `multiply(A, B; tolerance, maxbonddim = typemax(Int), final_truncation = true,
  nsamples = 4000, rng = Random.default_rng(), pilot_cutoff = tolerance^2 / (N - 1)^2,
  safety = 0.5, max_attempts = 3) -> (C, info)`
- `multiply(A, B; cutoff, maxbonddim = typemax(Int), final_truncation = true,
  nsamples = 0, rng = Random.default_rng()) -> (C, info)`
- `MultiplyInfo`: `peak_bonddim`, `peak_bonddim_per_step`, `bonddims`, `n_swaps`,
  `discarded_weight`, `hit_maxbonddim`, `error_bound`, `error_estimate`, `error_stderr`,
  `verified`, `attempts`, `cutoffs`, `nsamples`.

Threading is BLAS threading; the swap SVDs dominate and are LAPACK-bound.

## Related

`AlternatingCrossInterpolation.jl` interpolates the product instead of forming it; it wins
where the product is much simpler than its factors and loses where it is not. `BubbleTeaCI`
exposes this package as `contract(...; alg = "ttm")`.
