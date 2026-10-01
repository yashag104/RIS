%WEEK3_MISMATCH Robustness of the physics-aware estimator to element-model error.
%   Truth uses the Table I parameters; the receiver builds B from perturbed ones:
%   both resonances shifted by df0 (0, 5, 10, 20 MHz) and decay rates scaled by
%   independent (1 + U[-0.1, 0.1]) factors. Also the ideal +/-1 model (worst case).
%   Np = 2 pilots + 2 decision-directed iterations (16-QAM at 20 dB, 64-QAM at 30 dB).
%   Pass if df0 <= 10 MHz costs < 1 dB extra (rate-equivalent) at 20 dB.

clear; rng(800);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8; D = 0.75;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
[F, lamF] = dpss_basis(bins, df, 0.3e-6, 16);
shifts = [0 5e6 10e6 20e6]; snrs = [20 30]; qam = [16 64]; ntrial = 10; Np = 2; Nd = 6;
R = struct('df0_hz', shifts, 'snr_db', snrs);
for sc = {'S1', 'S2'}
  sc = sc{1};
  rp = zeros(1, 2); re = zeros(numel(shifts) + 1, 2);
  for t = 1:ntrial
    if strcmp(sc, 'S1'), seqs = repmat(base, Q, 1); groups = {1:Q}; mode = 'linear';
    else, seqs = zeros(Q,K); for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
         groups = num2cell(1:Q); mode = 'rank1';
    end
    [As, Ad] = phys_channel(Q, bins, df, fc, popt);
    H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
    models = cell(1, numel(shifts) + 1);
    for i = 1:numel(shifts)
        Pm = P; Pm.f0 = P.f0 + shifts(i);
        if shifts(i) > 0
            Pm.xr = P.xr .* (1 + 0.2*(rand(1,2) - 0.5)); Pm.xi = P.xi .* (1 + 0.2*(rand(1,2) - 0.5));
        end
        models{i} = Pm;
    end
    models{end} = 'ideal';
    Pc = mean(sum(abs(H(:, evalb)).^2, 1));
    for is = 1:2
        s2 = Pc / 10^(snrs(is)/10); Mq = qam(is);
        S = (sign(randn(M,Np)) + 1j*sign(randn(M,Np))) / sqrt(2);
        Yp = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
        Xd = qammod_(randi(Mq, M, Nd) - 1, Mq);
        Yd = H*Xd + sqrt(s2/2)*(randn(M,Nd) + 1j*randn(M,Nd));
        rp(is) = rp(is) + mmse_rate(H, H, s2, evalb);
        for im = 1:numel(models)
            Bset = cell(1, numel(groups));
            for g = 1:numel(groups)
                Bset{g} = coupling_matrix(bins, fc, df, seqs(groups{g}(1),:), ones(1,M), ones(1,M), models{im}, M-1);
            end
            Hh = est_lmmse(Yp, S, Bset, F, lamF, s2, mode, 15);
            for it = 1:2
                Xh = qamdec_(((Hh'*Hh + s2*eye(M)) \ Hh') * Yd, Mq);
                Hh = est_lmmse([Yp Yd], [S Xh], Bset, F, lamF, s2, mode, 15);
            end
            re(im, is) = re(im, is) + mmse_rate(Hh, H, s2, evalb);
        end
    end
  end
  rp = rp / ntrial; re = re / ntrial;
  R.(sc).rate_perfect = rp;
  R.(sc).rate_est = re;            % rows: df0 = 0, 5, 10, 20 MHz, ideal model
  slope = [0.31 0.33];             % bit/s/Hz per dB near 20/30 dB (from gate3d perfect curves)
  R.(sc).loss_db = (rp - re) ./ slope;
  fprintf('%s: perfect %s\n', sc, sprintf('%.2f ', rp));
  lbl = [arrayfun(@(s) sprintf('df0 %2d MHz', s/1e6), shifts, 'uni', 0), {'ideal +-1 '}];
  for im = 1:numel(lbl)
      fprintf('  %-10s rate %s | loss dB %s\n', lbl{im}, sprintf('%.2f ', re(im,:)), sprintf('%.2f ', R.(sc).loss_db(im,:)));
  end
end
fid = fopen(fullfile(out, 'week3_mismatch.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

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
x = reshape((q(real(z)*sc) + 1j*q(imag(z)*sc)) / sc, size(z));
end
