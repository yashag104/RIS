%WEEK3_FULLBAND_ERR Where does the full-band estimation gap come from? (S1, 28 dB)
%   Operator NMSE ||(He - H) x||^2 / ||H x||^2 over random x, matrix-free, for:
%   pilots only (Np = 2), genie extra pilots (Np = 6 true), and basis settings.
%   Separates estimator/basis error from decision-error propagation.

clear; rng(950);
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 3276; bins = -M/2:M/2-1;
Q = 16; K = 8; D = 0.75; snr_db = 28;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
seqs = repmat(base, Q, 1);
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
[As, Ad] = phys_channel(Q, bins, df, fc, popt);
op = fast_coupling(bins, fc, df, seqs, As, Ad, P);
Bop = {fast_coupling(bins, fc, df, seqs(1,:), ones(1,M), ones(1,M), P)};
xt = randn(M, 8) + 1j*randn(M, 8); Hx = op.H(xt);
s2 = mean(abs(Hx(:)).^2) / 2 / 10^(snr_db/10);       % per-sample SNR ~ 28 dB (x has power 2)
s2 = s2 * 2;                                           % unit-power symbols below
settings = {0.2e-6, 10};   % tau, extra L beyond M*df*tau
for is = 1:size(settings, 1)
    tau = settings{is,1}; L = ceil(M*df*tau) + settings{is,2};
    [F, lamF] = dpss_basis(bins, df, tau, L);
    for Np = [2 6]
        S = (sign(randn(M,Np)) + 1j*sign(randn(M,Np))) / sqrt(2);
        Yp = op.H(S) + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
        [Ae, Se] = estimate_s1(Yp, S, Bop, F, lamF, s2, Q);
        ope = fast_coupling(bins, fc, df, seqs, Se, Ae, P);
        e = ope.H(xt) - Hx;
        fprintf('tau %.1f us, L %2d, Np %d: full-C  operator NMSE %.1f dB\n', tau*1e6, L, Np, 10*log10(sum(abs(e(:)).^2)/sum(abs(Hx(:)).^2)));
        for r = [6 16]
            [Ae, Se] = estimate_lowrank(Yp, S, Bop{1}, F, lamF, s2, r, Ae, Se, Q);
            ope = fast_coupling(bins, fc, df, seqs, Se, Ae, P);
            e = ope.H(xt) - Hx;
            fprintf('                       rank-%-2d operator NMSE %.1f dB\n', r, 10*log10(sum(abs(e(:)).^2)/sum(abs(Hx(:)).^2)));
            [Ae, Se] = estimate_s1(Yp, S, Bop, F, lamF, s2, Q);   % reset init for next r
        end
    end
end

function [Ae, Se] = estimate_s1(Y, S, Bop, F, lam, s2, Q)
[M, Np] = size(S); L = size(F, 2); y = Y(:);
pl = kron(lam(:), lam(:));
Phi = zeros(M*Np, L*L);
for p = 1:Np
    Gp = Bop{1}.H(S(:,p) .* F);
    for i = 1:L, Phi((p-1)*M+(1:M), (i-1)*L+(1:L)) = F(:, i) .* Gp; end
end
sc0 = max(real(y'*y)/numel(y) - s2, eps) / max(real(sum(abs(Phi).^2, 1) * pl)/numel(y), eps);
Dw = sqrt(sc0*pl);
c = Dw .* (((Dw .* (Phi'*Phi)) .* Dw.' + s2*eye(L*L)) \ (Dw .* (Phi'*y)));
C = reshape(c, L, L).';
[U, Sg, V] = svd(C); r = min(Q, L);
Ae = zeros(Q, M); Se = zeros(Q, M);
for k = 1:r
    Ae(k, :) = (F * (U(:,k) * sqrt(Sg(k,k)))).';
    Se(k, :) = (F * (conj(V(:,k)) * sqrt(Sg(k,k)))).';
end
end

function [Ae, Se] = estimate_lowrank(Y, S, Bo, F, lam, s2, r, Ae0, Se0, Q)
% C = U V.' with U, V in C^{L x r}; per-hop DPSS prior; ALS from the full-C SVD init.
[M, Np] = size(S); L = size(F, 2); y = Y(:);
U = (F' * Ae0(1:r, :).');                          % project init onto basis (L x r)
V = (F' * Se0(1:r, :).');
G = cell(1, Np); for p = 1:Np, G{p} = Bo.H(S(:,p) .* F); end   % B (S_p .* F), M x L
wl = repmat(lam(:), r, 1);
sc = mean(abs([U(:); V(:)]).^2 ./ [wl; wl]);        % prior scale from the init
Dh = sqrt(sc*wl);
for it = 1:15
    Pu = zeros(M*Np, L*r);
    for p = 1:Np, GV = G{p} * V;
        for k = 1:r, Pu((p-1)*M+(1:M), (k-1)*L+(1:L)) = F .* GV(:, k); end
    end
    U = reshape(Dh .* (((Dh .* (Pu'*Pu)) .* Dh.' + s2*eye(L*r)) \ (Dh .* (Pu'*y))), L, r);
    Pv = zeros(M*Np, L*r); FU = F*U;
    for p = 1:Np
        for k = 1:r, Pv((p-1)*M+(1:M), (k-1)*L+(1:L)) = FU(:, k) .* G{p}; end
    end
    V = reshape(Dh .* (((Dh .* (Pv'*Pv)) .* Dh.' + s2*eye(L*r)) \ (Dh .* (Pv'*y))), L, r);
end
Ae = zeros(Q, M); Se = zeros(Q, M);
Ae(1:r, :) = (F*U).'; Se(1:r, :) = (F*V).';
end
