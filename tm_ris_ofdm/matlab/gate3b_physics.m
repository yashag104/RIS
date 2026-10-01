%GATE3B_PHYSICS Week 2, second estimator: physics-aware (known control sequence).
%   Gate 3 showed the model-agnostic per-harmonic estimator floors at -14..-16 dB
%   NMSE: rectangular switching leaves a 1/h^2 harmonic tail that no truncated
%   harmonic set captures. The legitimate receiver can know the sequences and the
%   element response family, so every harmonic is tied by a known matrix:
%
%     H(mbar, m) = sum_q B_q(mbar, m) * Ad_q(mbar) * As_q(m)
%
%   with B_q the coupling of element q for unit channels. Per hop the channels
%   are delay-limited, so Ad_q = Fd*a_q, As_q = Fs*s_q and
%     S1 (identical sequences):  H = B o (Fd * C * Fs.'),  C = sum_q a_q s_q.'  (Ld x Ls unknowns)
%     S2 (per-element sequences): H = sum_q B_q o (Fd * C_q * Fs.')             (Q*Ld*Ls unknowns)
%
%   Same pass criterion as Gate 3 (fixed before either run): with Np <= 2 the
%   rate is within ~1 dB of perfect-H MMSE and clearly above the baselines.
%   Also: B built from the IDEAL +/-1 model, to measure what modelling dispersion is worth.
%   Truth now keeps ALL in-band harmonics (hmax = M-1).

clear; rng(4);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8;
Np_list = [1 2 4 8 16];
snr_list = [10 20 30];
ntrial = 15;
tau_hop = 0.5e-6;                                     % assumed max delay per hop
dly = 0:1/(2*M*df):tau_hop; L = numel(dly);
F = exp(-1j*2*pi*bins(:)*df*dly);                      % M x L, same grid both hops
R = struct('Np', Np_list, 'snr_db', snr_list, 'L_per_hop', L);

for sc = {'S1', 'S2'}
  sc = sc{1}; D = 0.75;
  base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
  for isnr = 1:numel(snr_list)
    snr_db = snr_list(isnr);
    names = {'phys_exactB', 'phys_idealB', 'perfect'};
    rate = zeros(3, numel(Np_list)); nmse = rate;
    for t = 1:ntrial
        if strcmp(sc, 'S1'), seqs = repmat(base, Q, 1);
        else, seqs = zeros(Q,K); for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
        end
        As = tdl(Q, bins, df, 30e-9); Ad = tdl(Q, bins, df, 30e-9);
        H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
        % known per-element coupling for unit channels (exact and ideal models)
        Bx = cell(1, Q); Bi = cell(1, Q);
        uniq = strcmp(sc, 'S1');
        for q = 1:(uniq + ~uniq*Q)
            Bx{q} = coupling_matrix(bins, fc, df, seqs(q,:), ones(1,M), ones(1,M), P, M-1);
            Bi{q} = coupling_matrix(bins, fc, df, seqs(q,:), ones(1,M), ones(1,M), 'ideal', M-1);
        end
        Pc = mean(sum(abs(H(:, evalb)).^2, 1));
        s2 = Pc / 10^(snr_db/10);
        for ip = 1:numel(Np_list)
            Np = Np_list(ip);
            S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
            Y = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
            for e = 1:3
                if e == 3, Hh = H;
                else
                    if e == 1, Bset = Bx; else, Bset = Bi; end
                    nb = numel(find(~cellfun(@isempty, Bset)));
                    Phi = zeros(M*Np, nb*L*L);
                    for q = 1:nb
                        for p = 1:Np
                            G = Bset{q} * (S(:,p) .* F);              % M x L  (input side)
                            blk = zeros(M, L*L);
                            for i = 1:L, blk(:, (i-1)*L+(1:L)) = F(:, i) .* G; end
                            Phi((p-1)*M+(1:M), (q-1)*L*L+(1:L*L)) = blk;
                        end
                    end
                    lam = s2 * 1e-2;
                    c = (Phi'*Phi + lam*eye(size(Phi,2))) \ (Phi'*Y(:));
                    Hh = zeros(M);
                    for q = 1:nb
                        C = reshape(c((q-1)*L*L+(1:L*L)), L, L).';   % C(i, j): i output, j input
                        Hh = Hh + Bset{q} .* (F * C * F.');
                    end
                end
                Ee = Hh(evalb, evalb) - H(evalb, evalb);
                nmse(e, ip) = nmse(e, ip) + norm(Ee, 'fro')^2 / norm(H(evalb, evalb), 'fro')^2;
                W = (Hh'*Hh + s2*eye(M)) \ Hh';
                Gm = W * H; sig = abs(diag(Gm)).^2;
                sinr = sig ./ (sum(abs(Gm).^2, 2) - sig + s2*sum(abs(W).^2, 2));
                rate(e, ip) = rate(e, ip) + mean(log2(1 + sinr(evalb)));
            end
        end
    end
    key = sprintf('%s_snr%d', sc, snr_db);
    fprintf('%s:\n', key);
    for e = 1:3
        R.(key).(names{e}).rate = rate(e,:)/ntrial;
        R.(key).(names{e}).nmse_db = 10*log10(nmse(e,:)/ntrial);
        fprintf('  %-12s rate %s | nmse %s\n', names{e}, sprintf('%6.2f', rate(e,:)/ntrial), ...
            sprintf('%7.1f', 10*log10(nmse(e,:)/ntrial)));
    end
  end
end
fid = fopen(fullfile(out, 'gate3b.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function A = tdl(Q, bins, df, rms)
tau = (0:11) * rms / 2;
pdp = exp(-tau / rms); pdp = pdp / sum(pdp);
A = zeros(Q, numel(bins));
for q = 1:Q
    g = sqrt(pdp/2) .* (randn(1,12) + 1j*randn(1,12));
    A(q, :) = g * exp(-1j*2*pi*tau.' * (bins*df));
end
end
