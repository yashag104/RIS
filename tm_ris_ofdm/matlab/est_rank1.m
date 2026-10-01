function Hh = est_rank1(Y, S, Bset, F, rel_ridge, iters)
%EST_RANK1 Physics-aware bilinear estimator for per-element control sequences.
%   Each element's two-hop matrix is rank one, C_q = a_q * s_q.', so
%     H = sum_q Bset{q} o ((F a_q) (F s_q).')
%   with 2*L unknowns per element instead of L^2. Alternating least squares,
%   initialised from the rank-1 truncation of the linear estimate.

[M, Np] = size(S); L = size(F, 2); Q = numel(Bset);
% initialisation: linear estimate, then rank-1 per element
Phi = zeros(M*Np, Q*L*L);
for q = 1:Q
    for p = 1:Np
        Gp = Bset{q} * (S(:,p) .* F);
        for i = 1:L
            Phi((p-1)*M+(1:M), (q-1)*L*L+(i-1)*L+(1:L)) = F(:, i) .* Gp;
        end
    end
end
Aq = Phi'*Phi;
c = (Aq + 10*rel_ridge*real(trace(Aq))/size(Aq,1)*eye(size(Aq))) \ (Phi'*Y(:));
a = zeros(L, Q); s = zeros(L, Q);
for q = 1:Q
    C = reshape(c((q-1)*L*L+(1:L*L)), L, L).';
    [U, Sg, V] = svd(C);
    a(:, q) = U(:,1) * sqrt(Sg(1,1)); s(:, q) = conj(V(:,1)) * sqrt(Sg(1,1));
end
y = Y(:);
for it = 1:iters
    % update a (s fixed): column (q,i) = F(:,i) .* (B_q (S_p .* (F s_q)))
    Pa = zeros(M*Np, Q*L);
    for q = 1:Q
        for p = 1:Np
            v = Bset{q} * (S(:,p) .* (F*s(:,q)));
            Pa((p-1)*M+(1:M), (q-1)*L+(1:L)) = F .* v;
        end
    end
    a = reshape(rsolve(Pa, y, rel_ridge), L, Q);
    % update s (a fixed): column (q,j) = (F a_q) .* (B_q (S_p .* F(:,j)))
    Ps = zeros(M*Np, Q*L);
    for q = 1:Q
        fa = F*a(:,q);
        for p = 1:Np
            Ps((p-1)*M+(1:M), (q-1)*L+(1:L)) = fa .* (Bset{q} * (S(:,p) .* F));
        end
    end
    s = reshape(rsolve(Ps, y, rel_ridge), L, Q);
end
Hh = zeros(M);
for q = 1:Q
    Hh = Hh + Bset{q} .* ((F*a(:,q)) * (F*s(:,q)).');
end
end

function x = rsolve(A, y, rel)
G = A'*A;
x = (G + rel*real(trace(G))/size(G,1)*eye(size(G))) \ (A'*y);
end
