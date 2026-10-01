%GATE3E_DD Decision-directed refinement at the failing Gate-3 points (zero extra pilots).
%   Np = 2 pilot symbols -> LMMSE (DPSS prior) estimate -> MMSE-equalize Nd data
%   symbols -> hard decisions (64-QAM at 30 dB, 16-QAM at 20 dB) -> re-estimate
%   with pilots + decided data treated as known -> repeat.
%   Decision errors are included (no genie). Same calibrated setting as gate3d_lmmse.

clear; rng(300);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8; D = 0.75;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
[F, lamF] = dpss_basis(bins, df, 0.3e-6, 16);
Np = 2; Nd = 6; nit = 2; ntrial = 12;
cases = {'iidTDL','S1'; 'iidTDL','G4'; 'iidTDL','S2'; 'phys','G4'; 'phys','S2'};
snrs = [20 30]; qam = [16 64];
R = struct();
for ic = 1:size(cases,1)
  for isn = 1:2
    snr_db = snrs(isn); Mq = qam(isn);
    acc = zeros(1, 2 + nit); ser = zeros(1, nit);
    for t = 1:ntrial
        [H, Bset, mode] = make_case(cases{ic,1}, cases{ic,2}, base, Q, K, bins, df, fc, P, popt, M);
        Pc = mean(sum(abs(H(:, evalb)).^2, 1)); s2 = Pc / 10^(snr_db/10);
        S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
        Yp = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
        Xd = qammod_(randi(Mq, M, Nd) - 1, Mq);
        Yd = H*Xd + sqrt(s2/2)*(randn(M,Nd) + 1j*randn(M,Nd));
        Hh = est_lmmse(Yp, S, Bset, F, lamF, s2, mode, 15);
        acc(1) = acc(1) + mmse_rate(H, H, s2, evalb);
        acc(2) = acc(2) + mmse_rate(Hh, H, s2, evalb);
        for it = 1:nit
            W = (Hh'*Hh + s2*eye(M)) \ Hh';
            Xh = qamdec_(W*Yd, Mq);
            ser(it) = ser(it) + mean(Xh(evalb,:) ~= Xd(evalb,:), 'all');
            Hh = est_lmmse([Yp Yd], [S Xh], Bset, F, lamF, s2, mode, 15);
            acc(2+it) = acc(2+it) + mmse_rate(Hh, H, s2, evalb);
        end
    end
    acc = acc / ntrial; ser = ser / ntrial;
    key = sprintf('%s_%s_snr%d', cases{ic,1}, cases{ic,2}, snr_db);
    R.(key) = struct('rate_perfect', acc(1), 'rate_pilot_only', acc(2), 'rate_dd', acc(3:end), 'ser_per_iter', ser);
    fprintf('%s: perfect %.2f | pilots-only %.2f | DD iters %s | SER %s\n', key, acc(1), acc(2), ...
        sprintf('%.2f ', acc(3:end)), sprintf('%.1e ', ser));
  end
end
fid = fopen(fullfile(out, 'gate3e_dd.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function [H, Bset, mode] = make_case(ch, sc, base, Q, K, bins, df, fc, P, popt, M)
switch sc
    case 'S1', seqs = repmat(base, Q, 1); groups = {1:Q};
    case 'G4', seqs = zeros(Q,K); groups = cell(1,4);
        for g = 1:4, sh = circshift(base, randi(K)-1); idx = (g-1)*4+(1:4);
            seqs(idx,:) = repmat(sh, 4, 1); groups{g} = idx; end
    case 'S2', seqs = zeros(Q,K); groups = num2cell(1:Q);
        for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
end
if strcmp(ch, 'iidTDL'), As = tdl(Q,bins,df,30e-9); Ad = tdl(Q,bins,df,30e-9);
else, [As, Ad] = phys_channel(Q, bins, df, fc, popt); end
H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
Bset = cell(1, numel(groups));
for g = 1:numel(groups)
    Bset{g} = coupling_matrix(bins, fc, df, seqs(groups{g}(1),:), ones(1,M), ones(1,M), P, M-1);
end
mode = 'linear'; if strcmp(sc, 'S2'), mode = 'rank1'; end
end

function r = mmse_rate(Hh, H, s2, evalb)
M = size(H, 1);
W = (Hh'*Hh + s2*eye(M)) \ Hh'; G = W*H; sg = abs(diag(G)).^2;
sinr = sg ./ (sum(abs(G).^2, 2) - sg + s2*sum(abs(W).^2, 2));
r = mean(log2(1 + sinr(evalb)));
end

function x = qammod_(d, Mq)
m = sqrt(Mq); lv = -(m-1):2:(m-1);
x = (lv(mod(d, m) + 1) + 1j*lv(floor(d / m) + 1)) / sqrt(2*(Mq-1)/3);
end

function x = qamdec_(z, Mq)
m = sqrt(Mq); sc = sqrt(2*(Mq-1)/3); lv = -(m-1):2:(m-1);
q = @(v) lv(min(max(round((v + m - 1)/2), 0), m-1) + 1);
x = (q(real(z)*sc) + 1j*q(imag(z)*sc)) / sc;
x = reshape(x, size(z));
end

function A = tdl(Q, bins, df, rms)
tau = (0:11) * rms / 2;
pdp = exp(-tau / rms); pdp = pdp / sum(pdp);
A = zeros(Q, numel(bins));
for q = 1:Q
    g = sqrt(pdp/2) .* (randn(1,12) + 1j*randn(1,12));
    A(q, :) = g * exp(-1j*2*pi*tau.' * (bins*df));
end
end
