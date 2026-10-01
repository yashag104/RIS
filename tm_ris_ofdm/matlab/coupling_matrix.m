function H = coupling_matrix(m, fc, df, seqs, As, Ad, P, hmax)
%COUPLING_MATRIX Subcarrier-domain channel through a time-modulated RIS.
%   y(mbar) = sum_m H(mbar, m) s(m),  with (Iudice et al., eq. 25, scaling dropped)
%   H(mbar, m) = sum_q Ad(q, mbar) * b_q^[mbar - m](fc + m*df) * As(q, m)
%
%   m     : active subcarrier indices (vector, consecutive integers)
%   seqs  : Q x K control sequences, one row per element (states in {1,2})
%   As,Ad : Q x numel(m) per-element source->element and element->destination
%           frequency responses on the active grid
%   hmax  : harmonics |h| <= hmax kept (coupling to inactive bins is dropped)

m = m(:).';
M = numel(m);
Q = size(seqs, 1);
fin = fc + m*df;
H = zeros(M, M);
hs = -hmax:hmax;
for q = 1:Q
    B = harmonic_coeffs(fin, seqs(q, :), hs, P);   % (2hmax+1) x M, column = input m
    for ih = 1:numel(hs)
        h = hs(ih);
        src = find(m + h >= m(1) & m + h <= m(end));   % inputs whose output stays active
        dst = src + h;
        idx = sub2ind([M M], dst, src);
        H(idx) = H(idx) + Ad(q, dst) .* B(ih, src) .* As(q, src);
    end
end
end
