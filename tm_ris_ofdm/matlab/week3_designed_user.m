%WEEK3_DESIGNED_USER Fair comparison: TM configuration designed for the legitimate user.
%   Each element picks a polarity (base sequence or its state-complement), which
%   flips the sign of its h = 0 coefficient, chosen to co-phase the h = 0
%   (unshifted) component at the legitimate user (1-bit alignment, greedy over
%   the common reference). Per-element random cyclic shifts keep the
%   harmonics scrambled elsewhere (TM-IRS directional-modulation style).
%   This is the configuration most favourable to the conventional one-tap (TIN)
%   receiver, so it bounds our gain from below.
%   Reported at the designed user and at a random-polarity (generic) user:
%   TIN rate, perfect-H MMSE, estimated-H MMSE (Np = 2, LMMSE rank-1).

clear; rng(400);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192; c0 = M/2 + 1;
Q = 16; K = 8;
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
[F, lamF] = dpss_basis(bins, df, 0.3e-6, 16);
duties = [0.625 0.75 0.875]; snrs = [10 20 30]; ntrial = 10; Np = 2;
R = struct('duties', duties, 'snr_db', snrs);
for id = 1:numel(duties)
  D = duties(id);
  base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
  for user = {'designed', 'generic'}
    user = user{1};
    rt = zeros(1, numel(snrs)); rm = rt; re = rt; sir = 0; dom = 0;
    for t = 1:ntrial
        [As, Ad] = phys_channel(Q, bins, df, fc, popt);
        shifts = randi(K, Q, 1) - 1;
        cq = Ad(:, c0) .* As(:, c0);                     % per-element cascade at band centre
        b0 = harmonic_coeffs(fc, base, 0, P);           % h = 0 coefficient of the base sequence
        if strcmp(user, 'designed')
            best = -inf;
            for psi = linspace(0, 2*pi, 33)
                p = real(cq * b0 * exp(-1j*psi)) >= 0;   % keep base if it adds, else complement
                v = abs(sum(cq .* b0 .* (2*p - 1)));
                if v > best, best = v; pol = p; end
            end
        else
            pol = rand(Q, 1) > 0.5;
        end
        seqs = zeros(Q, K);
        for q = 1:Q
            s = circshift(base, shifts(q));
            if ~pol(q), s = 3 - s; end                   % complement: swap states 1 <-> 2
            seqs(q, :) = s;
        end
        H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
        Bset = cell(1, Q);
        for q = 1:Q
            Bset{q} = coupling_matrix(bins, fc, df, seqs(q,:), ones(1,M), ones(1,M), P, M-1);
        end
        dg = abs(diag(H)).^2; tot = sum(abs(H).^2, 1).';
        sir = sir + 10*log10(mean(dg(evalb)) / mean(tot(evalb) - dg(evalb)));
        dom = dom + mean(dg(evalb) ./ tot(evalb));
        Pc = mean(tot(evalb));
        for is = 1:numel(snrs)
            s2 = Pc / 10^(snrs(is)/10);
            sinr_t = dg ./ (sum(abs(H).^2, 2) - dg + s2);
            rt(is) = rt(is) + mean(log2(1 + sinr_t(evalb)));
            rm(is) = rm(is) + mmse_rate(H, H, s2, evalb);
            S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
            Y = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
            Hh = est_lmmse(Y, S, Bset, F, lamF, s2, 'rank1', 15);
            re(is) = re(is) + mmse_rate(Hh, H, s2, evalb);
        end
    end
    key = sprintf('D%03d_%s', round(1000*D), user);
    R.(key) = struct('rate_tin', rt/ntrial, 'rate_mmse_perfect', rm/ntrial, ...
        'rate_mmse_est_Np2', re/ntrial, 'sir_db', sir/ntrial, 'h0_energy_fraction', dom/ntrial);
    fprintf('%s: SIR %.1f dB, h0 fraction %.2f | TIN %s | MMSE %s | MMSE-est %s\n', key, sir/ntrial, dom/ntrial, ...
        sprintf('%5.2f ', rt/ntrial), sprintf('%5.2f ', rm/ntrial), sprintf('%5.2f ', re/ntrial));
  end
end
fid = fopen(fullfile(out, 'week3_designed_user.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function r = mmse_rate(Hh, H, s2, evalb)
M = size(H, 1);
W = (Hh'*Hh + s2*eye(M)) \ Hh'; G = W*H; sg = abs(diag(G)).^2;
sinr = sg ./ (sum(abs(G).^2, 2) - sg + s2*sum(abs(W).^2, 2));
r = mean(log2(1 + sinr(evalb)));
end
