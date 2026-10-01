%WEEK3_FAST_EQ Exact fast operator + conjugate-gradient MMSE equalizer.
%   (1) Verify fast_coupling against the explicit matrix (H x and H' y).
%   (2) CG solution of (H'H + s2 I) x = H' y with the fast operator, for
%       perfect and estimated channels: BER vs CG iterations against direct
%       full MMSE. (3) Flop/time scaling with M (256..2048).

clear; rng(600);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
Q = 16; K = 8; D = 0.75; Mq = 16;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
R = struct();

%% (1) exactness
M = 256; bins = -M/2:M/2-1;
seqs = zeros(Q,K); for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
[As, Ad] = phys_channel(Q, bins, df, fc, popt);
H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
op = fast_coupling(bins, fc, df, seqs, As, Ad, P);
x = randn(M,3) + 1j*randn(M,3);
R.exact.rel_err_Hx = norm(op.H(x) - H*x) / norm(H*x);
R.exact.rel_err_Hty = norm(op.Ht(x) - H'*x) / norm(H'*x);
fprintf('fast operator rel. error: Hx %.2e, H''y %.2e\n', R.exact.rel_err_Hx, R.exact.rel_err_Hty);

%% (2) BER vs CG iterations (S2, physical channel, 20 and 24 dB)
evalb = 65:192; Nd = 20; iters = [5 10 20 40]; snrs = [20 24]; ntrial = 8;
[F, lamF] = dpss_basis(bins, df, 0.3e-6, 16);
err = zeros(numel(snrs), 2 + 2*numel(iters)); nb = 0;
for t = 1:ntrial
    for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
    [As, Ad] = phys_channel(Q, bins, df, fc, popt);
    H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
    op = fast_coupling(bins, fc, df, seqs, As, Ad, P);
    Bset = cell(1,Q);
    for q = 1:Q, Bset{q} = coupling_matrix(bins, fc, df, seqs(q,:), ones(1,M), ones(1,M), P, M-1); end
    Pc = mean(sum(abs(H(:, evalb)).^2, 1));
    d = randi(Mq, M, Nd) - 1; X = qammod_(d, Mq); ref = d(evalb, :);
    for is = 1:numel(snrs)
        s2 = Pc / 10^(snrs(is)/10);
        Y = H*X + sqrt(s2/2)*(randn(M,Nd) + 1j*randn(M,Nd));
        % estimated channel -> per-element (a_q, s_q) -> fast operator on the estimate
        S = (sign(randn(M,2)) + 1j*sign(randn(M,2))) / sqrt(2);
        Yp = H*S + sqrt(s2/2)*(randn(M,2) + 1j*randn(M,2));
        [~, Ae, Se] = est_lmmse_factors(Yp, S, Bset, F, lamF, s2, 15);
        ope = fast_coupling(bins, fc, df, seqs, Se, Ae, P);
        zd = ((H'*H + s2*eye(M)) \ H') * Y;
        err(is, 1) = err(is, 1) + sum(qamdemod_(zd(evalb,:), Mq) ~= ref, 'all');
        He = zeros(M); for q = 1:Q, He = He + Bset{q} .* (Ae(q,:).' * Se(q,:)); end
        ze = ((He'*He + s2*eye(M)) \ He') * Y;
        err(is, 2) = err(is, 2) + sum(qamdemod_(ze(evalb,:), Mq) ~= ref, 'all');
        for ii = 1:numel(iters)
            zc = cg_mmse(op, Y, s2, iters(ii));
            err(is, 2+ii) = err(is, 2+ii) + sum(qamdemod_(zc(evalb,:), Mq) ~= ref, 'all');
            zc = cg_mmse(ope, Y, s2, iters(ii));
            err(is, 2+numel(iters)+ii) = err(is, 2+numel(iters)+ii) + sum(qamdemod_(zc(evalb,:), Mq) ~= ref, 'all');
        end
    end
    nb = nb + numel(ref);
end
R.ser.snr_db = snrs; R.ser.cg_iters = iters;
R.ser.direct_perfect = err(:,1).'/nb; R.ser.direct_est = err(:,2).'/nb;
R.ser.cg_perfect = err(:, 3:2+numel(iters))/nb; R.ser.cg_est = err(:, 3+numel(iters):end)/nb;
fprintf('SER direct perfect %s | direct est %s\n', mat2str(R.ser.direct_perfect,3), mat2str(R.ser.direct_est,3));
for ii = 1:numel(iters)
    fprintf('  CG %2d iters: perfect %s | est %s\n', iters(ii), mat2str(R.ser.cg_perfect(:,ii).',3), mat2str(R.ser.cg_est(:,ii).',3));
end

%% (3) run time vs M: direct MMSE (one solve) vs CG (20 iters), one OFDM symbol
Ms = [256 512 1024 2048]; tdir = nan(size(Ms)); tcg = tdir;
for im = 1:numel(Ms)
    M = Ms(im); bins = -M/2:M/2-1;
    [As, Ad] = phys_channel(Q, bins, df, fc, popt);
    op = fast_coupling(bins, fc, df, seqs, As, Ad, P);
    y = op.H(randn(M,1) + 1j*randn(M,1));
    tic; zc = cg_mmse(op, y, 1e-2, 20); tcg(im) = toc; %#ok<NASGU>
    if M <= 1024
        H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
        tic; zd = (H'*H + 1e-2*eye(M)) \ (H'*y); tdir(im) = toc; %#ok<NASGU>
    end
end
R.timing = struct('M', Ms, 'direct_s', tdir, 'cg20_s', tcg);
fprintf('timing (s): M %s | direct %s | CG20 %s\n', mat2str(Ms), mat2str(tdir,3), mat2str(tcg,3));
fid = fopen(fullfile(out, 'week3_fast_eq.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function z = cg_mmse(op, Y, s2, nit)
% Conjugate gradient on (H'H + s2 I) z = H'Y, column by column (all columns at once).
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
