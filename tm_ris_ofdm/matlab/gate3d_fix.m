%GATE3D_FIX Close the two Gate-3 gaps with one fixed setting, no per-SNR tuning.
%   Criterion (unchanged from Gate 3): with Np <= 2 pilot symbols, the rate of
%   the MMSE equalizer built from the estimate is within ~1 dB (rate-equivalent
%   SNR) of perfect-H MMSE, at 10, 20 and 30 dB.
%
%   Protocol against tuning-on-test: phase A picks (tau_max, L, ridge) on a
%   calibration seed at the hardest point (30 dB, Np = 2, S1); phase B
%   evaluates that single setting on fresh seeds everywhere.
%
%   Cases: channel in {iid-TDL (Iudice general form), physical (shared paths)}
%          x sequences in {S1 identical, G4 row groups, S2 per-element}
%   Estimators: linear (G*L^2 unknowns) for S1/G4; rank-1 ALS for S2.

clear;
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8; D = 0.75;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
R = struct();

%% Phase A: calibration (seed 100), S1, both channels, 30 dB, Np = 2
rng(100);
taus = [0.3e-6 0.5e-6]; Ls = [5 6 7 8]; ridges = [1e-4 1e-3 1e-2];
best = struct('loss', inf);
B1 = {coupling_matrix(bins, fc, df, base, ones(1,M), ones(1,M), P, M-1)};
trials = cell(1, 6);
for t = 1:6
    if t <= 3, [As, Ad] = deal(tdl(Q,bins,df,30e-9), tdl(Q,bins,df,30e-9));
    else, [As, Ad] = phys_channel(Q, bins, df, fc, popt); end
    H = coupling_matrix(bins, fc, df, repmat(base,Q,1), As, Ad, P, M-1);
    trials{t} = H;
end
for tm = taus
    for L = Ls
        F = dpss_basis(bins, df, tm, L);
        for rr = ridges
            loss = 0;
            for t = 1:6
                [rp, re] = eval_case(trials{t}, B1, F, rr, 30, 2, evalb, 'linear', rng);
                loss = loss + (rp - re);
            end
            loss = loss / 6;
            if loss < best.loss, best = struct('loss', loss, 'tau', tm, 'L', L, 'ridge', rr); end
        end
    end
end
R.calibration = best;
fprintf('calibrated: tau %.2g us, L %d, ridge %g (mean rate loss %.3f bit/s/Hz at 30 dB, Np 2)\n', ...
    best.tau*1e6, best.L, best.ridge, best.loss);

%% Phase B: fresh seeds, single setting
rng(200);
F = dpss_basis(bins, df, best.tau, best.L);
snrs = [10 20 30]; Nps = [1 2 4]; ntrial = 12;
chans = {'iidTDL', 'phys'}; seqcases = {'S1', 'G4', 'S2'};
for ic = 1:2
  for is = 1:3
    rp = zeros(numel(snrs), 1); re = zeros(numel(snrs), numel(Nps)); nm = re;
    for t = 1:ntrial
        switch seqcases{is}
            case 'S1', seqs = repmat(base, Q, 1); groups = {1:Q};
            case 'G4', seqs = zeros(Q,K); groups = cell(1,4);
                for g = 1:4, sh = circshift(base, randi(K)-1); idx = (g-1)*4+(1:4);
                    seqs(idx,:) = repmat(sh, 4, 1); groups{g} = idx; end
            case 'S2', seqs = zeros(Q,K); groups = num2cell(1:Q);
                for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
        end
        if ic == 1, As = tdl(Q,bins,df,30e-9); Ad = tdl(Q,bins,df,30e-9);
        else, [As, Ad] = phys_channel(Q, bins, df, fc, popt); end
        H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
        Bset = cell(1, numel(groups));
        for g = 1:numel(groups)
            Bset{g} = coupling_matrix(bins, fc, df, seqs(groups{g}(1),:), ones(1,M), ones(1,M), P, M-1);
        end
        mode = 'linear'; if strcmp(seqcases{is}, 'S2'), mode = 'rank1'; end
        for isn = 1:numel(snrs)
            for ip = 1:numel(Nps)
                [a1, a2, a3] = eval_case(H, Bset, F, best.ridge, snrs(isn), Nps(ip), evalb, mode, rng);
                if ip == 1, rp(isn) = rp(isn) + a1; end
                re(isn, ip) = re(isn, ip) + a2; nm(isn, ip) = nm(isn, ip) + a3;
            end
        end
    end
    key = sprintf('%s_%s', chans{ic}, seqcases{is});
    R.(key).snr_db = snrs; R.(key).Np = Nps;
    R.(key).rate_perfect = (rp/ntrial).';
    R.(key).rate_est = re/ntrial;
    R.(key).nmse_db = 10*log10(nm/ntrial);
    % rate-equivalent SNR loss: at high SNR ~3 dB per bit; use local slope of perfect curve
    slope = gradient(R.(key).rate_perfect, snrs);         % bit/s/Hz per dB
    R.(key).loss_db = (R.(key).rate_perfect.' - R.(key).rate_est) ./ slope.';
    fprintf('%s: perfect %s\n', key, sprintf('%6.2f', R.(key).rate_perfect));
    for ip = 1:numel(Nps)
        fprintf('   Np %d: rate %s  loss(dB) %s  nmse %s\n', Nps(ip), sprintf('%6.2f', R.(key).rate_est(:,ip)), ...
            sprintf('%6.2f', R.(key).loss_db(:,ip)), sprintf('%7.1f', R.(key).nmse_db(:,ip)));
    end
  end
end
fid = fopen(fullfile(out, 'gate3d.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function [rp, re, nm] = eval_case(H, Bset, F, rr, snr_db, Np, evalb, mode, ~)
M = size(H, 1);
Pc = mean(sum(abs(H(:, evalb)).^2, 1)); s2 = Pc / 10^(snr_db/10);
S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
Y = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
if strcmp(mode, 'linear'), Hh = est_linear(Y, S, Bset, F, rr);
else, Hh = est_rank1(Y, S, Bset, F, rr, 15); end
rp = mmse_rate(H, H, s2, evalb); re = mmse_rate(Hh, H, s2, evalb);
E = Hh(evalb, evalb) - H(evalb, evalb);
nm = norm(E, 'fro')^2 / norm(H(evalb, evalb), 'fro')^2;
end

function r = mmse_rate(Hh, H, s2, evalb)
M = size(H, 1);
W = (Hh'*Hh + s2*eye(M)) \ Hh'; G = W*H; sg = abs(diag(G)).^2;
sinr = sg ./ (sum(abs(G).^2, 2) - sg + s2*sum(abs(W).^2, 2));
r = mean(log2(1 + sinr(evalb)));
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
