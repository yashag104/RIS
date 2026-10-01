%WEEK3_BANDED_BER Uncoded BER Monte Carlo and a low-complexity banded equalizer.
%   Receivers (16-QAM, physical channel, Q = 16, K = 8, D = 0.75):
%     tin        one tap per subcarrier (conventional OFDM)
%     full       full M x M MMSE, perfect H
%     band_W     sliding-window MMSE per subcarrier m: observations m-W..m+W,
%                interferers m-2W..m+2W, perfect H. Cost O(M W^3) vs O(M^3).
%     full_est   full MMSE with H estimated from Np = 2 pilots (LMMSE, DPSS prior)
%                + 2 decision-directed iterations on the frame's data symbols
%   Scenarios: S1 (identical sequences) and S2 (per-element random shifts).

clear; rng(500);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8; D = 0.75; Mq = 16;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
popt = struct('P', 5, 'K_db', 10, 'rms', 30e-9);
[F, lamF] = dpss_basis(bins, df, 0.3e-6, 16);
snrs = 8:4:28; Ws = [2 4 8 16]; ntrial = 10; Nd = 20; Np = 2;
R = struct('snr_db', snrs, 'W', Ws, 'qam', Mq);
for sc = {'S1', 'S2'}
  sc = sc{1};
  names = [{'tin', 'full', 'full_est'}, arrayfun(@(w) sprintf('band_%d', w), Ws, 'uni', 0)];
  err = zeros(numel(names), numel(snrs)); nbits = 0;
  for t = 1:ntrial
    if strcmp(sc, 'S1'), seqs = repmat(base, Q, 1); groups = {1:Q}; mode = 'linear';
    else, seqs = zeros(Q,K); for q = 1:Q, seqs(q,:) = circshift(base, randi(K)-1); end
         groups = num2cell(1:Q); mode = 'rank1';
    end
    [As, Ad] = phys_channel(Q, bins, df, fc, popt);
    H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
    Bset = cell(1, numel(groups));
    for g = 1:numel(groups)
        Bset{g} = coupling_matrix(bins, fc, df, seqs(groups{g}(1),:), ones(1,M), ones(1,M), P, M-1);
    end
    Pc = mean(sum(abs(H(:, evalb)).^2, 1));
    d = randi(Mq, M, Nd) - 1; X = qammod_(d, Mq);
    bits_ref = de2bi_(d(evalb, :), log2(Mq));
    for is = 1:numel(snrs)
        s2 = Pc / 10^(snrs(is)/10);
        Y = H*X + sqrt(s2/2)*(randn(M,Nd) + 1j*randn(M,Nd));
        S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
        Yp = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
        Z = cell(1, numel(names));
        Z{1} = Y ./ diag(H);
        Z{2} = ((H'*H + s2*eye(M)) \ H') * Y;
        Hh = est_lmmse(Yp, S, Bset, F, lamF, s2, mode, 15);
        for it = 1:2
            Xh = qamdec_(((Hh'*Hh + s2*eye(M)) \ Hh') * Y, Mq);
            Hh = est_lmmse([Yp Y], [S Xh], Bset, F, lamF, s2, mode, 15);
        end
        Z{3} = ((Hh'*Hh + s2*eye(M)) \ Hh') * Y;
        for iw = 1:numel(Ws)
            Z{3+iw} = band_mmse(H, Y, s2, Ws(iw));
        end
        for e = 1:numel(names)
            dh = qamdemod_(Z{e}(evalb, :), Mq);
            err(e, is) = err(e, is) + sum(de2bi_(dh, log2(Mq)) ~= bits_ref, 'all');
        end
    end
    nbits = nbits + numel(bits_ref);
  end
  for e = 1:numel(names)
      R.(sc).(names{e}) = err(e, :) / nbits;
  end
  fprintf('%s BER vs SNR %s dB:\n', sc, mat2str(snrs));
  for e = 1:numel(names), fprintf('  %-9s %s\n', names{e}, sprintf('%9.2e', err(e,:)/nbits)); end
end
fid = fopen(fullfile(out, 'week3_banded_ber.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function Z = band_mmse(H, Y, s2, W)
% Per-subcarrier sliding-window MMSE: rows m-W..m+W, columns m-2W..m+2W.
M = size(H, 1); Z = zeros(size(Y));
for m = 1:M
    r = max(1, m-W):min(M, m+W);
    c = max(1, m-2*W):min(M, m+2*W);
    Hw = H(r, c);
    k = find(c == m);
    w = (Hw*Hw' + s2*eye(numel(r))) \ Hw(:, k);
    w = w / real(w' * Hw(:, k));                      % unbiased
    Z(m, :) = w' * Y(r, :);
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

function b = de2bi_(d, nb)
% Gray-coded bits per axis (natural index -> Gray), nb/2 bits per axis.
m = 2^(nb/2);
ir = mod(d, m); iq = floor(d / m);
g = @(v) bitxor(v, bitshift(v, -1));
b = false([size(d), nb]);
gi = g(ir); gq = g(iq);
for k = 1:nb/2
    b(:,:,k) = bitget(gi, k); b(:,:,nb/2+k) = bitget(gq, k);
end
end
