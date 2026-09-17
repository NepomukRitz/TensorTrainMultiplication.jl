# TensorTrainMultiplication.jl

The elementwise (Hadamard) product of two tensor trains through the swap network of
Michailidis, Fenton and Kiffner, *Tensor Train Multiplication*, arXiv:2410.19747.

```julia
using TensorTrainMultiplication
C, info = multiply(A, B; cutoff = 1e-10)
info.peak_bonddim, info.error_bound
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
tight accuracy targets; the memory advantage holds regardless.

## Accuracy

`cutoff` is the relative discarded squared weight allowed at one SVD (the paper's `epsilon`,
ITensors' relative `cutoff`). It says nothing directly about the error of the product, which
accumulates over `N (N - 1) / 2` swaps and a final sweep; see **Choosing a cutoff** below.

What the result does report is `info.error_bound = sum_i sqrt(w_i)`, summed over every
truncation the pass performed. It is a rigorous upper bound on the relative L2 error of the
returned product: each truncation happens in mixed-canonical gauge, so its error is exactly
`sqrt(w_i)` times the current norm, and the triangle inequality sums them. Being a triangle
inequality over many terms it is also pessimistic, by a factor that grows with the number of
truncating swaps: about 2x to 16x on 12-site products, about 8x on a 28-site production
object. Use it as a certificate that an error is small enough, never as an estimate of how
large it is.

The bound covers this multiplication's own truncations. Errors the inputs already carry are
inherited by any product and are not visible here.

## Choosing a cutoff

For an algorithm that re-truncates each bond many times -- this swap network, and a
variational fit alike -- the relative L2 error behaves as

```
err ~ kappa sqrt(cutoff)
```

with `kappa` measured at 1.2 to 2.6 times the number of bonds on a 28-site production object.
To target a relative L2 error `tol`, take

```
cutoff = (tol / (2 n_bonds))^2
```

the factor 2 being the margin.

This is a calibration, not a theorem. It was measured on one object, and `kappa` is a
property of the trains being multiplied, not of the algorithm alone; on a different product
it can sit outside that range, and a target met on one problem can be missed on another.
Treat the rule as a starting cutoff, then read `info.error_bound` to see whether the pass
that ran actually stayed inside the target.

The single-sweep case is where a theorem exists, and the contrast is the point. One rounding
sweep truncates each bond exactly once, the individual errors are orthogonal and add in
quadrature, and `cutoff = tol^2 / n_bonds` therefore *guarantees* a relative error at or
below `tol` (Oseledets' TT-rounding bound). That is `n_bonds` times looser a cutoff than the
rule above. Repeated re-truncation of the same bond is what destroys the quadrature argument:
the errors are no longer orthogonal, they can add linearly, and the safe cutoff picks up the
second power of the bond count.

## API

- `multiply(A, B; cutoff = 0.0, maxbonddim = typemax(Int), final_truncation = true) -> (C, info)`
- `MultiplyInfo`: `peak_bonddim`, `peak_bonddim_per_step`, `bonddims`, `n_swaps`,
  `discarded_weight`, `hit_maxbonddim`, `error_bound`.

`cutoff = 0.0` keeps everything above the numerical noise floor and returns the exact
product. `maxbonddim` caps every bond the algorithm creates and sets `info.hit_maxbonddim`
when it removes anything; a capped product is returned, not thrown, and `info.error_bound`
covers what the cap discarded. Threading is BLAS threading; the swap SVDs dominate and are
LAPACK-bound.

## Related

`AlternatingCrossInterpolation.jl` interpolates the product instead of forming it; it wins
where the product is much simpler than its factors and loses where it is not.
