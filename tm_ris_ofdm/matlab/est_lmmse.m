function Hh = est_lmmse(Y, S, Bset, F, lam, s2, mode, iters)
%EST_LMMSE Bayesian (LMMSE) physics-aware estimator with a DPSS prior.
%   Each hop's delay-domain coefficients get prior variance proportional to
%   the DPSS eigenvalues lam (delays uniform on [0, tau_max]), so a larger
%   basis no longer amplifies noise. Prior scale is fitted from the data.
%   mode 'linear': H = sum_g B_g o (F C_g F.'),  vec(C_g) ~ CN(0, sc * kron(lam, lam))
%   mode 'rank1' : H = sum_q B_q o ((F a_q)(F s_q).'), a_q, s_q ~ CN(0, sc_h * lam)

[M, Np] = size(S); L = size(F, 2); G = numel(Bset); y = Y(:);
pl = kron(lam(:), lam(:));                       % prior shape of vec(C): index (i-1)*L + j
Phi = zeros(M*Np, G*L*L);
for g = 1:G
    for p = 1:Np
        Gp = Bset{g} * (S(:,p) .* F);
        for i = 1:L
            Phi((p-1)*M+(1:M), (g-1)*L*L+(i-1)*L+(1:L)) = F(:, i) .* Gp;
        end
    end
end
A = Phi'*Phi; b = Phi'*y; w = repmat(pl, G, 1);
% prior scale: match the expected received energy (per sample, minus noise)
sc = max(real(y'*y)/numel(y) - s2, eps) / max(real(sum(sum(abs(Phi).^2, 1).' .* w))/numel(y), eps);
Dw = sqrt(sc*w);                                 % whitened (numerically stable) LMMSE
c = Dw .* ((Dw.*A.*Dw.' + s2*eye(numel(w))) \ (Dw .* b));
if strcmp(mode, 'linear')
    Hh = zeros(M);
    for g = 1:G
        C = reshape(c((g-1)*L*L+(1:L*L)), L, L).';
        Hh = Hh + Bset{g} .* (F * C * F.');
    end
    return
end
% rank-1 refinement (per element), prior on each hop
a = zeros(L, G); s = zeros(L, G);
for q = 1:G
    C = reshape(c((q-1)*L*L+(1:L*L)), L, L).';
    [U, Sg, V] = svd(C); a(:,q) = U(:,1)*sqrt(Sg(1,1)); s(:,q) = conj(V(:,1))*sqrt(Sg(1,1));
end
sh = sqrt(sc);                                    % per-hop prior scale
wl = repmat(lam(:), G, 1);
for it = 1:iters
    Pa = zeros(M*Np, G*L);
    for q = 1:G
        for p = 1:Np
            Pa((p-1)*M+(1:M), (q-1)*L+(1:L)) = F .* (Bset{q} * (S(:,p) .* (F*s(:,q))));
        end
    end
    Dh = sqrt(sh*wl);
    a = reshape(Dh .* (((Dh.*(Pa'*Pa)).*Dh.' + s2*eye(numel(wl))) \ (Dh .* (Pa'*y))), L, G);
    Ps = zeros(M*Np, G*L);
    for q = 1:G
        fa = F*a(:,q);
        for p = 1:Np
            Ps((p-1)*M+(1:M), (q-1)*L+(1:L)) = fa .* (Bset{q} * (S(:,p) .* F));
        end
    end
    s = reshape(Dh .* (((Dh.*(Ps'*Ps)).*Dh.' + s2*eye(numel(wl))) \ (Dh .* (Ps'*y))), L, G);
end
Hh = zeros(M);
for q = 1:G
    Hh = Hh + Bset{q} .* ((F*a(:,q)) * (F*s(:,q)).');
end
end
