function [Hh, Ae, Se] = est_lmmse_factors(Y, S, Bset, F, lam, s2, iters)
%EST_LMMSE_FACTORS Rank-1 LMMSE estimate returning per-element hop responses.
[Hh, Ae, Se] = est_lmmse(Y, S, Bset, F, lam, s2, 'rank1', iters);
end
