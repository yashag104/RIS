function Hh = est_linear(Y, S, Bset, F, rel_ridge)
%EST_LINEAR Physics-aware linear estimator, one full delay matrix per sequence group.
%   H = sum_g Bset{g} o (F * C_g * F.'),  unknowns G*L^2, ridge relative to
%   the regressor power. Y, S: M x Np received / pilot symbols.

[M, Np] = size(S); L = size(F, 2); G = numel(Bset);
Phi = zeros(M*Np, G*L*L);
for g = 1:G
    for p = 1:Np
        Gp = Bset{g} * (S(:,p) .* F);                % M x L, input side
        cols = (g-1)*L*L;
        for i = 1:L
            Phi((p-1)*M+(1:M), cols+(i-1)*L+(1:L)) = F(:, i) .* Gp;
        end
    end
end
A = Phi'*Phi;
c = (A + rel_ridge*real(trace(A))/size(A,1)*eye(size(A))) \ (Phi'*Y(:));
Hh = zeros(M);
for g = 1:G
    C = reshape(c((g-1)*L*L+(1:L*L)), L, L).';
    Hh = Hh + Bset{g} .* (F * C * F.');
end
end
