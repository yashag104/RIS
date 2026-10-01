function B = harmonic_coeffs(f, seq, h, P)
%HARMONIC_COEFFS Fourier coefficients of a periodically switched dispersive element.
%   B(i, n) = b^[h(i)](f(n)) for a K-slot control sequence SEQ (state index per
%   slot, values in {1,2}), so that Gamma(f, t) = sum_h b^[h](f) exp(j 2 pi h t / T).
%
%   b^[h](f) = exp(-j pi h / K) / K * sinc(h / K) * sum_k Gamma_k(f) exp(-j 2 pi h k / K)
%   (Iudice et al., eq. 5, with k = 0..K-1 and normalized sinc).

K = numel(seq);
h = h(:);
f = f(:).';
k = (0:K-1);
G = zeros(K, numel(f));
for s = 1:K
    G(s, :) = element_response(f, seq(s), P);
end
W = exp(-1j*2*pi*h*k/K);                     % (H x K)
B = (exp(-1j*pi*h/K)/K .* snc(h/K)) .* (W*G);
end

function y = snc(x)
y = ones(size(x));
nz = x ~= 0;
y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end
