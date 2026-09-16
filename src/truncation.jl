"""
    truncated_rank(s, cutoff, maxbonddim) -> (k, discarded, capped)

How many of the descending singular values `s` to keep.

The rule is ITensors' relative `cutoff`, which is `epsilon` of arXiv:2410.19747, Eq. 10: the
longest tail whose summed squares stay at or below `cutoff` times the total sum of squares is
dropped; values at or below the numerical noise floor `length(s) * eps * s[1]` are dropped as
well; at least one value is kept. `k` is then capped at `maxbonddim`. `discarded` is the
relative squared weight removed by both rules together, `capped` whether the cap removed
anything.
"""
function truncated_rank(s::AbstractVector{<:Real}, cutoff::Real, maxbonddim::Integer)
    n = length(s)
    n == 0 && return 0, 0.0, false
    total = sum(abs2, s)
    total > 0 || return 1, 0.0, false
    noise = n * eps(float(eltype(s))) * s[1]
    k = n
    tail = 0.0
    while k > 1
        next = tail + abs2(s[k])
        (next <= cutoff * total || s[k] <= noise) || break
        tail = next
        k -= 1
    end
    capped = k > maxbonddim
    if capped
        for j in (maxbonddim + 1):k
            tail += abs2(s[j])
        end
        k = maxbonddim
    end
    return k, tail / total, capped
end
