%WEEK3_SEQ_DESIGN Channel-aware shift selection for the legitimate user's conditioning.
%   Per-element cyclic shifts (S2 style) degrade the conditioning of H.
%   Greedy coordinate ascent: for each element in turn, try all K shifts and keep
%   the one that maximizes the legit user's MMSE rate (perfect H, 20 dB), 2 sweeps.
%   Also report an eavesdropper (independent channel draw, same sequences):
%   its one-tap SIR, i.e. whether scrambling toward others survives the design.

clear; rng(700);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8; D = 0.75; snr_db = 20; ntrial = 8;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
res = zeros(ntrial, 6);
for t = 1:ntrial
    [As, Ad] = phys_channel(Q, bins, df, fc, popt);       % legit user
    [Ae, De] = phys_channel(Q, bins, df, fc, popt);       % eavesdropper (other position)
    De = De .* exp(1j*2*pi*rand(Q,1));                    % different geometry phases
    sh = randi(K, Q, 1) - 1;
    Bq = cell(Q, K);                                      % precompute per-element, per-shift couplings
    for q = 1:Q
        for k = 1:K
            Bq{q,k} = coupling_matrix(bins, fc, df, circshift(base, k-1), As(q,:), Ad(q,:), P, M-1);
        end
    end
    Hs = @(sh) sum(cat(3, Bq{sub2ind([Q K], (1:Q).', sh+1)}), 3);
    H0 = Hs(sh); s2 = mean(sum(abs(H0(:, evalb)).^2, 1)) / 10^(snr_db/10);
    r0 = mmse_rate(H0, s2, evalb);
    for sweep = 1:2
        for q = 1:Q
            best = -inf;
            for k = 0:K-1
                sh2 = sh; sh2(q) = k; r = mmse_rate(Hs(sh2), s2, evalb);
                if r > best, best = r; kb = k; end
            end
            sh(q) = kb;
        end
    end
    H1 = Hs(sh); r1 = mmse_rate(H1, s2, evalb);
    Hid = sum(cat(3, Bq{:, 1}), 3); rid = mmse_rate(Hid, s2, evalb);   % all identical (no shifts)
    % eavesdropper one-tap SIR with random vs designed shifts
    seqr = zeros(Q,K); seqd = seqr;
    shr = randi(K, Q, 1) - 1;
    for q = 1:Q, seqr(q,:) = circshift(base, shr(q)); seqd(q,:) = circshift(base, sh(q)); end
    Er = coupling_matrix(bins, fc, df, seqr, Ae, De, P, M-1); Ed = coupling_matrix(bins, fc, df, seqd, Ae, De, P, M-1);
    sir = @(E) 10*log10(mean(abs(diag(E(evalb,evalb))).^2) / mean(sum(abs(E(evalb,:)).^2,2) - abs(diag(E(evalb,evalb))).^2));
    res(t, :) = [r0, r1, rid, sir(Er), sir(Ed), numel(unique(sh))];
    fprintf('trial %d: legit MMSE rate random %.2f -> designed %.2f (identical %.2f) | eve SIR random %.1f dB, designed %.1f dB | distinct shifts %d\n', t, res(t,:));
end
R = struct('rate_random', mean(res(:,1)), 'rate_designed', mean(res(:,2)), 'rate_identical', mean(res(:,3)), ...
    'eve_sir_random_db', mean(res(:,4)), 'eve_sir_designed_db', mean(res(:,5)), 'distinct_shifts', mean(res(:,6)), 'snr_db', snr_db);
disp(R);
fid = fopen(fullfile(out, 'week3_seq_design.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function r = mmse_rate(H, s2, evalb)
M = size(H, 1);
W = (H'*H + s2*eye(M)) \ H'; G = W*H; sg = abs(diag(G)).^2;
sinr = sg ./ (sum(abs(G).^2, 2) - sg + s2*sum(abs(W).^2, 2));
r = mean(log2(1 + sinr(evalb)));
end
