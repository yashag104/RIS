%WEEK3_FULLBAND_DIAG (diagnostic: CG iterations and DD passes at 28 dB) Full n78 band (M = 3276, 98.3 MHz), matrix-free throughout.
%   Estimation regressors and equalization use only the exact FFT operator
%   (FAST_COUPLING); no M x M matrix is formed. Physical channel, Q = 16, K = 8,
%   D = 0.75, Np = 2 pilots + 1 decision-directed iteration.
%   Receivers: one-tap (TIN), CG-MMSE with true channel, CG-MMSE with estimate.
%   Metric: uncoded SER on the central half of the band.

clear; rng(900);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 3276; bins = -M/2:M/2-1; evalb = round(M/4)+1:round(3*M/4);
Q = 16; K = 8; D = 0.75;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
tau = 0.3e-6; L = ceil(M*df*tau) + 6;
tic; [F, lamF] = dpss_basis(bins, df, tau, L); tb = toc;
fprintf('M %d, L %d per hop, DPSS basis %.1f s\n', M, L, tb);
cfg = {'S1', 28, 64; 'S2', 28, 64};
Np = 2; Nd = 8; ntrial = 2; nit_cg = 80; n_dd = 3;
R = struct('M', M, 'L', L);
for ic = 1:size(cfg, 1)
    sc = cfg{ic,1}; snr_db = cfg{ic,2}; Mq = cfg{ic,3};
    errs = zeros(1, 3); nsym = 0; ttot = 0;
    for t = 1:ntrial
        if strcmp(sc, 'S1'), seqs = repmat(base, Q, 1);
        else, seqs = zeros(Q,K); for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
        end
        [As, Ad] = phys_channel(Q, bins, df, fc, popt);
        op = fast_coupling(bins, fc, df, seqs, As, Ad, P);
        % unit-channel operators per distinct sequence (the receiver's known B)
        if strcmp(sc, 'S1'), Bop = {fast_coupling(bins, fc, df, seqs(1,:), ones(1,M), ones(1,M), P)};
        else, Bop = cell(1,Q); for q = 1:Q, Bop{q} = fast_coupling(bins, fc, df, seqs(q,:), ones(1,M), ones(1,M), P); end
        end
        % noise from the mean collected energy (power method-free: sum of |H e_m|^2 over a few columns)
        cols = evalb(round(linspace(1, numel(evalb), 24)));
        E = zeros(M, numel(cols)); E(sub2ind(size(E), cols, 1:numel(cols))) = 1;
        Pc = mean(sum(abs(op.H(E)).^2, 1)); s2 = Pc / 10^(snr_db/10);
        S = (sign(randn(M,Np)) + 1j*sign(randn(M,Np))) / sqrt(2);
        Yp = op.H(S) + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
        d = randi(Mq, M, Nd) - 1; X = qammod_(d, Mq);
        Yd = op.H(X) + sqrt(s2/2)*(randn(M,Nd) + 1j*randn(M,Nd));
        tic;
        [Ae, Se] = estimate(Yp, S, Bop, F, lamF, s2, sc, Q);
        ope = fast_coupling(bins, fc, df, seqs, Se, Ae, P);
        for dd = 1:n_dd
            Xh = qamdec_(cg_mmse(ope, Yd, s2, nit_cg), Mq);
            [Ae, Se] = estimate([Yp Yd], [S Xh], Bop, F, lamF, s2, sc, Q);
            ope = fast_coupling(bins, fc, df, seqs, Se, Ae, P);
        end
        ttot = ttot + toc;
        % receivers
        diagH = op.H(eye(M, 1)); %#ok<NASGU>
        dg = onetap_diag(op, M);
        z = {Yd ./ dg, cg_mmse(op, Yd, s2, nit_cg), cg_mmse(ope, Yd, s2, nit_cg)};
        for e = 1:3
            errs(e) = errs(e) + sum(qamdemod_(z{e}(evalb,:), Mq) ~= d(evalb,:), 'all');
        end
        nsym = nsym + numel(evalb)*Nd;
    end
    key = sprintf('%s_snr%d_qam%d', sc, snr_db, Mq);
    R.(key) = struct('ser_tin', errs(1)/nsym, 'ser_cg_true', errs(2)/nsym, 'ser_cg_est', errs(3)/nsym, ...
        'est_time_s', ttot/ntrial);
    fprintf('%s: SER one-tap %.3g | CG true %.3g | CG est %.3g | est+DD time %.1f s\n', key, ...
        errs(1)/nsym, errs(2)/nsym, errs(3)/nsym, ttot/ntrial);
end
fid = fopen(fullfile(out, 'week3_fullband_diag.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function dg = onetap_diag(op, M)
% Diagonal of H via probing with combs: columns m, m+step, ... (coupling decays,
% so leakage between comb teeth is negligible only approximately; use exact
% per-column probing in blocks instead).
dg = zeros(M, 1); blk = 256;
for s = 1:blk:M
    idx = s:min(M, s+blk-1);
    E = zeros(M, numel(idx)); E(sub2ind(size(E), idx, 1:numel(idx))) = 1;
    Hc = op.H(E);
    dg(idx) = Hc(sub2ind(size(Hc), idx, 1:numel(idx)));
end
end

function [Ae, Se] = estimate(Y, S, Bop, F, lam, s2, sc, Q)
% Matrix-free regressors: B (S_p .* F) via the fast operator.
[M, Np] = size(S); L = size(F, 2); y = Y(:);
if strcmp(sc, 'S1')
    pl = kron(lam(:), lam(:));
    Phi = zeros(M*Np, L*L);
    for p = 1:Np
        Gp = Bop{1}.H(S(:,p) .* F);
        for i = 1:L, Phi((p-1)*M+(1:M), (i-1)*L+(1:L)) = F(:, i) .* Gp; end
    end
    sc0 = max(real(y'*y)/numel(y) - s2, eps) / max(real(sum(abs(Phi).^2, 1) * pl)/numel(y), eps);
    Dw = sqrt(sc0*pl);
    c = Dw .* (((Dw .* (Phi'*Phi)) .* Dw.' + s2*eye(L*L)) \ (Dw .* (Phi'*y)));
    C = reshape(c, L, L).';                        % C(i,j): output i, input j
    % represent sum_q a_q s_q^T = C exactly with Q "virtual elements" via SVD
    [U, Sg, V] = svd(C); r = min(Q, L);
    Ae = zeros(Q, M); Se = zeros(Q, M);
    for k = 1:r
        Ae(k, :) = (F * (U(:,k) * sqrt(Sg(k,k)))).';
        Se(k, :) = (F * (conj(V(:,k)) * sqrt(Sg(k,k)))).';
    end
    return
end
% S2: rank-1 ALS per element, init from a common two-hop matrix with mean B
wl = repmat(lam(:), Q, 1);
Gq = cell(Q, Np);
for q = 1:Q, for p = 1:Np, Gq{q,p} = Bop{q}.H(S(:,p) .* F); end, end
Phi = zeros(M*Np, L*L); pl = kron(lam(:), lam(:));
for p = 1:Np
    Gm = zeros(M, L); for q = 1:Q, Gm = Gm + Gq{q,p}; end
    for i = 1:L, Phi((p-1)*M+(1:M), (i-1)*L+(1:L)) = F(:, i) .* Gm; end
end
sc0 = max(real(y'*y)/numel(y) - s2, eps) / max(real(sum(abs(Phi).^2, 1) * pl)/numel(y), eps);
Dw = sqrt(sc0*pl);
c = Dw .* (((Dw .* (Phi'*Phi)) .* Dw.' + s2*eye(L*L)) \ (Dw .* (Phi'*y)));
[U, Sg, V] = svd(reshape(c, L, L).');
a = repmat(U(:,1)*sqrt(Sg(1,1)), 1, Q); s = repmat(conj(V(:,1))*sqrt(Sg(1,1)), 1, Q);
sh = sqrt(sc0) * Q;                                % per-element prior scale
Dh = sqrt(sh*wl);
for it = 1:20
    Pa = zeros(M*Np, Q*L);
    for q = 1:Q, for p = 1:Np
        Pa((p-1)*M+(1:M), (q-1)*L+(1:L)) = F .* (Gq{q,p} * s(:,q));
    end, end
    a = reshape(Dh .* (((Dh .* (Pa'*Pa)) .* Dh.' + s2*eye(Q*L)) \ (Dh .* (Pa'*y))), L, Q);
    Ps = zeros(M*Np, Q*L);
    for q = 1:Q, fa = F*a(:,q); for p = 1:Np
        Ps((p-1)*M+(1:M), (q-1)*L+(1:L)) = fa .* Gq{q,p};
    end, end
    s = reshape(Dh .* (((Dh .* (Ps'*Ps)) .* Dh.' + s2*eye(Q*L)) \ (Dh .* (Ps'*y))), L, Q);
end
Ae = (F*a).'; Se = (F*s).';
end

function z = cg_mmse(op, Y, s2, nit)
b = op.Ht(Y); z = zeros(size(b)); r = b; p = r; rs = sum(conj(r).*r, 1);
for it = 1:nit
    Ap = op.Ht(op.H(p)) + s2*p;
    al = rs ./ sum(conj(p).*Ap, 1);
    z = z + al.*p; r = r - al.*Ap;
    rn = sum(conj(r).*r, 1);
    p = r + (rn./rs).*p; rs = rn;
end
end

function x = qammod_(d, Mq)
m = sqrt(Mq); lv = -(m-1):2:(m-1);
x = (lv(mod(d, m) + 1) + 1j*lv(floor(d / m) + 1)) / sqrt(2*(Mq-1)/3);
end
function d = qamdemod_(z, Mq)
m = sqrt(Mq); sc = sqrt(2*(Mq-1)/3);
ir = min(max(round((real(z)*sc + m - 1)/2), 0), m-1);
iq = min(max(round((imag(z)*sc + m - 1)/2), 0), m-1);
d = ir + m*iq;
end
function x = qamdec_(z, Mq)
x = qammod_(qamdemod_(z, Mq), Mq);
end
