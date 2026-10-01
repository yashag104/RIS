%GATE3C_TUNE Basis / regularization sweep for the physics-aware S1 estimator.
%   Gate 3b: with Np = 1 there are 256 equations for 64 unknowns, yet NMSE was
%   only -11.5 dB at 30 dB SNR. Suspected cause: 2x-oversampled delay grid plus a
%   near-zero ridge, i.e. noise amplification. Sweep grid oversampling and a
%   ridge scaled to the regressor power. Same truth, same criterion as Gate 3.

clear; rng(5);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1; evalb = 65:192;
Q = 16; K = 8; D = 0.75;
base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
seqs = repmat(base, Q, 1);
B = coupling_matrix(bins, fc, df, base, ones(1,M), ones(1,M), P, M-1);
tau_hop = 0.5e-6; ntrial = 10;
osr_list = [1 2]; rel_ridge = [1e-4 1e-3 1e-2 1e-1];
snr_list = [20 30]; Np_list = [1 2];
R = struct();
for osr = osr_list
  dly = 0:1/(osr*M*df):tau_hop; L = numel(dly);
  F = exp(-1j*2*pi*bins(:)*df*dly);
  for rr = rel_ridge
    for snr_db = snr_list
      rate = zeros(1, numel(Np_list)); nm = rate; rp = 0;
      for t = 1:ntrial
        As = tdl(Q, bins, df, 30e-9); Ad = tdl(Q, bins, df, 30e-9);
        H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, M-1);
        Pc = mean(sum(abs(H(:, evalb)).^2, 1)); s2 = Pc / 10^(snr_db/10);
        Wp = (H'*H + s2*eye(M)) \ H'; Gp = Wp*H; sg = abs(diag(Gp)).^2;
        rp = rp + mean(log2(1 + sg(evalb)./(sum(abs(Gp(evalb,:)).^2,2) - sg(evalb) + s2*sum(abs(Wp(evalb,:)).^2,2))));
        for ip = 1:numel(Np_list)
          Np = Np_list(ip);
          S = (sign(randn(M, Np)) + 1j*sign(randn(M, Np))) / sqrt(2);
          Y = H*S + sqrt(s2/2)*(randn(M,Np) + 1j*randn(M,Np));
          Phi = zeros(M*Np, L*L);
          for p = 1:Np
            G = B * (S(:,p) .* F);
            for i = 1:L, Phi((p-1)*M+(1:M), (i-1)*L+(1:L)) = F(:, i) .* G; end
          end
          A = Phi'*Phi;
          lam = rr * real(trace(A)) / size(A,1);
          c = (A + lam*eye(size(A))) \ (Phi'*Y(:));
          Hh = B .* (F * reshape(c, L, L).' * F.');
          Ee = Hh(evalb, evalb) - H(evalb, evalb);
          nm(ip) = nm(ip) + norm(Ee,'fro')^2 / norm(H(evalb,evalb),'fro')^2;
          W = (Hh'*Hh + s2*eye(M)) \ Hh'; Gm = W*H; sg = abs(diag(Gm)).^2;
          sinr = sg ./ (sum(abs(Gm).^2,2) - sg + s2*sum(abs(W).^2,2));
          rate(ip) = rate(ip) + mean(log2(1 + sinr(evalb)));
        end
      end
      key = sprintf('osr%d_ridge%g_snr%d', osr, rr, snr_db); key = strrep(key, '.', 'p'); key = strrep(key, '-', 'm');
      R.(key) = struct('L', L, 'rate', rate/ntrial, 'nmse_db', 10*log10(nm/ntrial), 'perfect', rp/ntrial);
      fprintf('osr %d ridge %-6g snr %d (L=%d): rate Np1 %.2f Np2 %.2f | perfect %.2f | nmse %.1f %.1f\n', ...
        osr, rr, snr_db, L, rate/ntrial, rp/ntrial, 10*log10(nm/ntrial));
    end
  end
end
fid = fopen(fullfile(out, 'gate3c.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

function A = tdl(Q, bins, df, rms)
tau = (0:11) * rms / 2;
pdp = exp(-tau / rms); pdp = pdp / sum(pdp);
A = zeros(Q, numel(bins));
for q = 1:Q
    g = sqrt(pdp/2) .* (randn(1,12) + 1j*randn(1,12));
    A(q, :) = g * exp(-1j*2*pi*tau.' * (bins*df));
end
end
