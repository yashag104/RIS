%GATE2_EQUALIZATION Days 3-4: joint equalization vs treat-ICI-as-noise (perfect H).
%   Pre-registered prediction (written before running):
%     * TIN receiver (one tap per subcarrier, everything else is interference)
%       saturates at SIR = |H_mm|^2 / sum_{m' ~= m} |H_mm'|^2.
%     * Linear MMSE joint equalizer approaches the matched-filter bound
%       MF_m = ||H(:,m)||^2 / sigma^2 (all harmonic energy collected) when H
%       is well conditioned. Energy-collection gain = MF / (|H_mm|^2/sigma^2).
%     * For one group of identical binary elements with duty D (ideal model):
%       |b0|^2 = (2D-1)^2 and sum_h |b_h|^2 = 1, so the gain is 1/(2D-1)^2.
%   Kill criterion: MMSE does not track the MF bound, or the gain over TIN is
%   < 2 dB (rate-equivalent SNR) at realistic SNR (10-25 dB).
%
%   Scenarios (K = 8 slots, Q = 16 elements, dispersive OpenRIS cell):
%     S1  all elements use the same duty-D sequence (no space-time coding)
%     S2  each element uses a random cyclic shift of the duty-D sequence (TM-IRS style)
%   Output: ../results/gate2.json, ../results/gate2_*.png

clear; rng(2);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
M = 256; bins = -M/2:M/2-1;
evalb = 65:192;                          % score central subcarriers only (edge effects)
Q = 16; K = 8;
duties = [0.5 0.625 0.75 0.875];
snr_db = 0:5:30;
ntrial = 30;
R = struct('snr_db', snr_db, 'duties', duties);

for sc = 1:2
    for id = 1:numel(duties)
        D = duties(id);
        base = [ones(1, round(D*K)), 2*ones(1, K - round(D*K))];
        rt = zeros(numel(snr_db), 1); rm = rt; rmf = rt; gain_lin = zeros(ntrial,1);
        for t = 1:ntrial
            if sc == 1
                seqs = repmat(base, Q, 1);
            else
                seqs = zeros(Q, K);
                for q = 1:Q, seqs(q, :) = circshift(base, randi(K) - 1); end
            end
            As = tdl(Q, bins, df, 30e-9); Ad = tdl(Q, bins, df, 30e-9);
            H = coupling_matrix(bins, fc, df, seqs, As, Ad, P, 40);
            colE = sum(abs(H).^2, 1).';              % energy collected per input
            dg = abs(diag(H)).^2;
            Ptot = mean(colE(evalb));                % normalize: SNR = mean collected energy / sigma^2
            gain_lin(t) = mean(colE(evalb)) / mean(dg(evalb));
            for is = 1:numel(snr_db)
                s2 = Ptot / 10^(snr_db(is)/10);
                sir_i = sum(abs(H).^2, 2) - dg;      % row interference for TIN
                sinr_tin = dg ./ (sir_i + s2);
                Ginv = inv(H'*H / s2 + eye(M));
                sinr_mmse = 1 ./ real(diag(Ginv)) - 1;
                mf = colE / s2;
                rt(is) = rt(is) + mean(log2(1 + sinr_tin(evalb)));
                rm(is) = rm(is) + mean(log2(1 + sinr_mmse(evalb)));
                rmf(is) = rmf(is) + mean(log2(1 + mf(evalb)));
            end
        end
        key = sprintf('S%d_D%03d', sc, round(1000*D));
        R.(key).rate_tin = rt.'/ntrial;
        R.(key).rate_mmse = rm.'/ntrial;
        R.(key).rate_mf_bound = rmf.'/ntrial;
        R.(key).energy_gain_db = 10*log10(mean(gain_lin));
        R.(key).predicted_ideal_gain_db = -10*log10((2*D - 1)^2);
        % rate-equivalent SNR gain of MMSE over TIN at 10, 20 dB
        for target = [10 20]
            r_t = interp1(snr_db, R.(key).rate_tin, target);
            r_m = interp1(snr_db, R.(key).rate_mmse, target);
            % SNR TIN would need to reach MMSE's rate (NaN if it never does)
            need = interp1(R.(key).rate_tin, snr_db, r_m, 'linear', NaN);
            R.(key).(sprintf('snr_gain_at_%ddB', target)) = need - target;
            R.(key).(sprintf('rate_gain_at_%ddB', target)) = r_m - r_t;
        end
        fprintf('%s: energy gain %.2f dB (ideal pred %.2f), rate@20dB TIN %.2f MMSE %.2f MF %.2f\n', ...
            key, R.(key).energy_gain_db, R.(key).predicted_ideal_gain_db, ...
            interp1(snr_db, R.(key).rate_tin, 20), interp1(snr_db, R.(key).rate_mmse, 20), ...
            interp1(snr_db, R.(key).rate_mf_bound, 20));
    end
end

fid = fopen(fullfile(out, 'gate2.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);

fig = figure('Visible', 'off', 'Position', [100 100 1000 380]);
for sc = 1:2
    subplot(1, 2, sc); hold on; grid on
    cols = lines(numel(duties));
    for id = 1:numel(duties)
        key = sprintf('S%d_D%03d', sc, round(1000*duties(id)));
        plot(snr_db, R.(key).rate_mmse, '-o', 'Color', cols(id,:));
        plot(snr_db, R.(key).rate_tin, '--', 'Color', cols(id,:));
        plot(snr_db, R.(key).rate_mf_bound, ':', 'Color', cols(id,:));
    end
    xlabel('SNR (dB, all harmonic energy)'); ylabel('rate (bit/s/Hz per subcarrier)')
    title(sprintf('S%d: solid MMSE, dashed TIN, dotted MF bound', sc))
end
exportgraphics(fig, fullfile(out, 'gate2_rates.png'), 'Resolution', 150); close(fig)

function A = tdl(Q, bins, df, rms)
tau = (0:11) * rms / 2;
pdp = exp(-tau / rms); pdp = pdp / sum(pdp);
A = zeros(Q, numel(bins));
for q = 1:Q
    g = sqrt(pdp/2) .* (randn(1,12) + 1j*randn(1,12));
    A(q, :) = g * exp(-1j*2*pi*tau.' * (bins*df));
end
end
