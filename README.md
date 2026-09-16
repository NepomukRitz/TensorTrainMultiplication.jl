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
applies it to the other with a variational fit: cost `O(N d chi^4)`, memory `O(chi^3)` for
the fit's environments. TTM never forms the operator. It concatenates the two trains into one
chain, swaps corresponding carriages next to each other with one truncated SVD per swap, and
contracts each pair through a COPY tensor as soon as it is adjacent: cost `O(N^2 d^3 chi^3)`,
memory `O(d^2 chi^2)`, in the largest *intermediate* bond dimension. That intermediate bond
grows above the input rank during the swaps (`info.peak_bonddim` reports it) and decides
whether TTM wins. On the paper's turbulence data at `N = 28` the growth was 2.4x the input
rank at a per-SVD cutoff of `1e-12` and 3.7-5x at `1e-16`, with runtime crossovers against
the conventional algorithm at bond dimensions 200, 500 and 1400 (arXiv:2410.19747, Fig. 3).
The growth is steep in the cutoff, so TTM pays off at moderate accuracy and loses at very
tight tolerances; the memory advantage holds regardless.

## Accuracy: a measured tolerance

A per-SVD cutoff says something about one truncation only. A many-truncation algorithm, TTM
or a fit, ends with a relative error that is a problem-dependent multiple of `sqrt(cutoff)`,
about the number of bonds for functions with sharp features. `multiply(A, B; tolerance)`
therefore measures instead of assuming: the exact residual `C(x) - A(x) B(x)` is sampled at
random points, which needs no reference computation, and the cutoff is tightened from a pilot
pass until the estimate plus two standard errors is below the tolerance. The result reports
`error_estimate`, `error_stderr` and `verified`; when a bond cap or the attempt limit prevents
verification the product is returned with `verified = false`, nothing is thrown and nothing is
silent. The measurement covers this multiplication's own truncations; errors already present
in the inputs are inherited by any product.

**What `verified` does and does not say.** It says the *sampled* residual is below the
tolerance at that sample size. A uniform sample cannot see a residual that lives on a
vanishing fraction of the index grid, so a product can be certified wrongly. Two guards
narrow that gap without closing it: `verified` also requires the sample to have carried the
error — an effective sample size `ess_num = (sum x)^2 / sum(x^2)` of at least `min_ess` — or
the rigorous `error_bound` to be inside the tolerance on its own, which needs no sampling.
`ess_den` is reported for the denominator but not gated on: a sharply peaked product is
heavy-tailed there by construction, and a small `ess_den` widens the estimate's uncertainty
by roughly `1 / sqrt(ess_den)` rather than invalidating it.

`error_bound = sum_i sqrt(w_i)` over the discarded weights is a rigorous bound (every
truncation happens in mixed-canonical gauge) and is pessimistic by a factor that grows with
the number of truncating swaps. `cutoff` mode runs one pass at a fixed per-SVD cutoff for
expert use and, with `nsamples > 0`, still reports the sampled error.

## API

- `multiply(A, B; tolerance, maxbonddim = typemax(Int), final_truncation = true, nsamples = 4000, rng, pilot_cutoff = tolerance^2 / (N - 1)^2, safety = 0.5, max_attempts = 3, min_ess = 30) -> (C, info)`
- `multiply(A, B; cutoff, maxbonddim = typemax(Int), final_truncation = true, nsamples = 0, rng) -> (C, info)`
- `MultiplyInfo`: `peak_bonddim`, `peak_bonddim_per_step`, `bonddims`, `n_swaps`,
  `discarded_weight`, `hit_maxbonddim`, `error_bound`, `error_estimate`, `error_stderr`,
  `verified`, `attempts`, `cutoffs`, `nsamples`, `ess_num`, `ess_den`, `min_ess`.

Threading is BLAS threading; the swap SVDs dominate and are LAPACK-bound.

## Related

`AlternatingCrossInterpolation.jl` interpolates the product instead of forming it; it wins
where the product is much simpler than its factors and loses where it is not.
