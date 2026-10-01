%GATE3_ESTIMATION Week 2: structured pilot-based estimation of the coupled channel.
%   Model: H = sum_{h in Hset} S_h * diag(g_h), S_h shifts input m to output m+h.
%   Under the generalized CP condition each g_h(m) is delay-limited (propagation
%   delay spread + RIS memory), so g_h = F * c_h with F an M x L delay basis.
%   Unknowns: |Hset| * L, independent of the element physics.
%
%   Estimators, all from Np full-band pilot OFDM symbols (known QPSK):
%     conv      conventional OFDM: diagonal LS y_m/s_m, one-tap equalizer
%     toeplitz  ideal-model structure: H(mbar, m) = t(mbar - m), |Hset| unknowns
%     banded    unstructured LS of every band entry (|Hset| unknowns per row)
%     struct    proposed delay-limited per-harmonic LS (ridge)
%     perfect   true H
%   Score: NMSE on the central block and the achievable rate of the linear MMSE
%   equalizer built from the estimate and applied to the TRUE channel.
%
%   Pre-registered pass criterion: with Np <= 2 the proposed estimator's rate is
%   within ~1 dB (rate-equivalent SNR) of perfect-H MMSE and clearly above all
%   baselines at equal Np.
%   Output: ../results/gate3.json, ../results/gate3_*.png

clear; rng(3);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8;
Np_list = [1 2 4 8 16];
snr_db = 20;
ntrial = 20;
tau_max = 1.0e-6;                         % assumed max delay (< NR CP 2.34 us at 30 kHz)
dly = 0:1/(2*M*df):tau_max;               % 2x oversampled delay grid
F = exp(-1j*2*pi*bins(:)*df*dly);         % M x L basis
cases = {'S1', 0.75; 'S2', 0.75; 'S1', 0.5};
Hsets = {-8:8, -12:12};
R = struct('Np', Np_list, 'snr_db', snr_db, 'L', numel(dly), 'tau_max', tau_max);

for ic = 1:size(cases, 1)
  for iH = 1:numel(Hsets)
    hset = Hsets{iH};
    sc = cases{ic,1}; D = cases{ic,2};
    base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
    names = {'conv', 'toeplitz', 'banded', 'struct', 'perfect'};
    rate = zeros(numel(names), numel(Np_list)); nmse = rate;
    for t = 1:ntrial
        if strcmp(sc, 'S1'), seqs = repmat(base, Q, 1);
        else, seqs = zeros(Q,K); for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
        end
        As = tdl(Q, bins, df, 30e-9); Ad = tdl(Q, bins, df, 30e-9);
        H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, 60);
        Pc = mean(sum(abs(H(:, evalb)).^2, 1));
        s2 = Pc / 10^(snr_db/10);
        for ip = 1:numel(Np_list)
            Np = Np_list(ip);
            S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
            Y = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
            Hh = cell(1, 5);
            % conventional: diagonal only
            Hh{1} = diag(mean(Y ./ S, 2));
            % Toeplitz: H(mbar,m) = t(mbar-m), h in hset
            Phi = zeros(M*Np, numel(hset));
            for k = 1:numel(hset), Phi(:, k) = reshape(shiftmat(M, hset(k))*S, [], 1); end
            tt = Phi \ Y(:);
            Hh{2} = zeros(M); for k = 1:numel(hset), Hh{2} = Hh{2} + tt(k)*shiftmat(M, hset(k)); end
            % banded unstructured: per row, unknown H(mbar, mbar-h)
            Hb = zeros(M);
            for r = 1:M
                cols = r - hset; ok = cols >= 1 & cols <= M; cols = cols(ok);
                A = S(cols, :).';                          % Np x ncols
                Hb(r, cols) = (pinv(A) * Y(r, :).').';
            end
            Hh{3} = Hb;
            % proposed: delay-limited per-harmonic, ridge LS
            L = numel(dly); Phi = zeros(M*Np, numel(hset)*L);
            for k = 1:numel(hset)
                Sh = shiftmat(M, hset(k));
                for p = 1:Np
                    Phi((p-1)*M+(1:M), (k-1)*L+(1:L)) = Sh * (S(:,p) .* F);
                end
            end
            lam = s2 * 1e-2;
            c = (Phi'*Phi + lam*eye(size(Phi,2))) \ (Phi'*Y(:));
            Hs = zeros(M);
            for k = 1:numel(hset)
                Hs = Hs + shiftmat(M, hset(k)) * diag(F * c((k-1)*L+(1:L)));
            end
            Hh{4} = Hs;
            Hh{5} = H;
            for e = 1:5
                Ee = Hh{e}(evalb, evalb) - H(evalb, evalb);
                nmse(e, ip) = nmse(e, ip) + norm(Ee, 'fro')^2 / norm(H(evalb, evalb), 'fro')^2;
                if e == 1
                    sinr = abs(diag(H)).^2 ./ (sum(abs(H).^2, 2) - abs(diag(H)).^2 + s2);  % one-tap = TIN
                    % mismatch of the one-tap estimate only rotates/scales: use TIN with estimated tap
                    g = diag(Hh{1}); w = conj(g) ./ abs(g).^2;
                    sig = abs(w .* diag(H)).^2;
                    intf = abs(w).^2 .* (sum(abs(H).^2, 2) - abs(diag(H)).^2) + abs(w).^2 * s2;
                    sinr = sig ./ intf;
                else
                    W = (Hh{e}'*Hh{e} + s2*eye(M)) \ Hh{e}';   % MMSE from estimate
                    G = W * H;
                    sig = abs(diag(G)).^2;
                    intf = sum(abs(G).^2, 2) - sig + s2 * sum(abs(W).^2, 2);
                    sinr = sig ./ intf;
                end
                rate(e, ip) = rate(e, ip) + mean(log2(1 + sinr(evalb)));
            end
        end
    end
    key = sprintf('%s_D%03d_H%d', sc, round(1000*D), max(hset));
    for e = 1:5
        R.(key).(names{e}).rate = rate(e,:)/ntrial;
        R.(key).(names{e}).nmse_db = 10*log10(nmse(e,:)/ntrial);
    end
    R.(key).unknowns_struct = numel(hset)*numel(dly);
    R.(key).unknowns_banded_per_row = numel(hset);
    fprintf('%s (unknowns struct %d):\n', key, numel(hset)*numel(dly));
    for e = 1:5
        fprintf('  %-8s rate %s | nmse %s\n', names{e}, sprintf('%6.2f', rate(e,:)/ntrial), ...
            sprintf('%7.1f', 10*log10(nmse(e,:)/ntrial)));
    end
  end
end

fid = fopen(fullfile(out, 'gate3.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function S = shiftmat(M, h)
% Maps input subcarrier m to output m+h (truncated to the active band).
S = diag(ones(M - abs(h), 1), -h);
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
