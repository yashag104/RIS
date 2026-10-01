function [F, lam] = dpss_basis(bins, df, tau_max, L)
%DPSS_BASIS Orthonormal basis for frequency responses with delays in [0, tau_max].
%   Top-L eigenvectors of R(n,m) = E_tau[exp(-j 2 pi (n-m) df tau)], tau ~ U[0, tau_max]
%   = exp(-j pi (n-m) df tau_max) * sinc((n-m) df tau_max).
%   These are the (modulated) discrete prolate spheroidal sequences: the
%   best L-dimensional subspace for delay-limited responses, with no grid
%   to fall off (unlike an oversampled DFT delay grid). Base MATLAB only.

n = bins(:);
d = n - n.';
x = d * df * tau_max;
s = ones(size(x)); nz = x ~= 0; s(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
Rm = exp(-1j*pi*x) .* s;
Rm = (Rm + Rm') / 2;
[V, E] = eig(Rm, 'vector');
[~, ix] = sort(real(E), 'descend');
F = V(:, ix(1:L));
lam = real(E(ix(1:L)));
lam = lam / sum(lam);                  % prior power per basis function
end
