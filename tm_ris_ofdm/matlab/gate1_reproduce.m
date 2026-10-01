%GATE1_REPRODUCE Days 1-2: reproduce Iudice et al. and validate the model.
%   (1) In-channel dispersion of the OpenRIS cell over 100 MHz (their Sec. IV-B):
%       expect differential phase 180 +/- ~6.5 deg, imbalance < 0.75 dB,
%       per-state phase variation ~47/50 deg, ripple ~0.6 dB.
%   (2) Coupling-matrix structure for K = 2 binary control (their Sec. IV-C):
%       odd harmonics with sinc(h/2) envelope; even harmonics vanish in the
%       ideal model and leak only through dispersion.
%   (3) Independent check: closed-form coupling matrix vs a direct time-domain
%       simulation of the switched element (filter -> gate -> FFT), with
%       multi-element sequences and multipath on both hops.
%   Outputs: ../results/gate1.json, ../results/gate1_*.png

clear; rng(1);
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'results');
P = params_openris();
fc = 3.594e9; df = 30e3;
R = struct();

%% (1) dispersion over the 100 MHz channel
m = -1638:1637;                         % 3276 active subcarriers = 98.28 MHz
f = fc + m*df;
G0 = element_response(f, 1, P); G1 = element_response(f, 2, P);
dphi = rad2deg(angle(G0 ./ G1));        % differential phase, wrapped to (-180,180]
dphi180 = mod(dphi, 360);               % around +180
R.disp.diff_phase_deg = [min(dphi180) max(dphi180)];
R.disp.imbalance_db = [min(20*log10(abs(G0)./abs(G1))) max(20*log10(abs(G0)./abs(G1)))];
R.disp.phase_var_deg = [rng_(unwrap(angle(G0))) rng_(unwrap(angle(G1)))]*180/pi;
R.disp.ripple_db = [rng_(20*log10(abs(G0))) rng_(20*log10(abs(G1)))];

fig = figure('Visible', 'off', 'Position', [100 100 900 320]);
subplot(1,3,1); plot((f-fc)/1e6, 20*log10(abs([G0;G1]))); grid on
xlabel('f - f_c (MHz)'); ylabel('|\Gamma| (dB)'); legend('c_0','c_1','Location','best'); title('Magnitude')
subplot(1,3,2); plot((f-fc)/1e6, rad2deg(unwrap(angle([G0;G1]),[],2))); grid on
xlabel('f - f_c (MHz)'); ylabel('phase (deg)'); title('Phase per state')
subplot(1,3,3); plot((f-fc)/1e6, dphi180); grid on
xlabel('f - f_c (MHz)'); ylabel('\angle\Gamma_0/\Gamma_1 (deg)'); title('Differential phase')
exportgraphics(fig, fullfile(out, 'gate1_dispersion.png'), 'Resolution', 150); close(fig)

%% (2) coupling-matrix structure, K = 2, one element, no propagation
m64 = -32:31;
H = coupling_matrix(m64, fc, df*1, [1 2], ones(1,64), ones(1,64), P, 63);
Hid = coupling_matrix(m64, fc, df, [1 2], ones(1,64), ones(1,64), 'ideal', 63);
c = 33;                                  % a central input subcarrier
hs = -10:10;
R.coupling.h = hs;
R.coupling.disp_db = 20*log10(abs(H(c+hs, c)).' + 1e-300);
R.coupling.ideal_db = 20*log10(abs(Hid(c+hs, c)).' + 1e-300);
% even-harmonic leakage relative to the h = +1 term, across the full band
Bfull = harmonic_coeffs(f, [1 2], [0 1 2], P);
R.coupling.dc_leak_db = [min(20*log10(abs(Bfull(1,:)./Bfull(2,:)))) max(20*log10(abs(Bfull(1,:)./Bfull(2,:))))];
R.coupling.h2_leak_db = max(20*log10(abs(Bfull(3,:)./Bfull(2,:)) + 1e-300));

fig = figure('Visible', 'off', 'Position', [100 100 800 320]);
subplot(1,2,1); imagesc(m64, m64, 20*log10(abs(H)+1e-6)); axis image; colorbar; caxis([-60 0])
xlabel('transmitted subcarrier m'); ylabel('received subcarrier'); title('|H| (dB), dispersive, K=2')
subplot(1,2,2); stem(hs, R.coupling.disp_db, 'filled'); hold on
stem(hs, max(R.coupling.ideal_db, -80), 'r'); grid on; ylim([-80 0])
xlabel('harmonic h'); ylabel('|b^{[h]}| (dB)'); legend('dispersive','ideal \pm1','Location','south')
title('Column through a central subcarrier')
exportgraphics(fig, fullfile(out, 'gate1_coupling.png'), 'Resolution', 150); close(fig)

%% (3) closed form vs time-domain simulation
Nfft = 4096;                             % 64x oversampling: sampled gates ~ continuous
M = 64; bins = -32:31;
Q = 4; K = 4;
errs = zeros(1, 20); errs_id = zeros(1, 20);
for trial = 1:20
    seqs = randi(2, Q, K);
    As = tdl_response(Q, bins, df, 30e-9); Ad = tdl_response(Q, bins, df, 30e-9);
    X = (sign(randn(M,1)) + 1j*sign(randn(M,1)))/sqrt(2);
    for model = 1:2
        if model == 1, PP = P; else, PP = 'ideal'; end
        Hc = coupling_matrix(bins, fc, df, seqs, As, Ad, PP, M-1);
        % time-domain output keeps only active bins, as the matrix does
        Ytd = td_simulate(X, bins, Nfft, fc, df, seqs, As, Ad, PP);
        e = norm(Ytd - Hc*X) / norm(Ytd);
        if model == 1, errs(trial) = e; else, errs_id(trial) = e; end
    end
end
R.td_check.rel_err_dispersive = [median(errs) max(errs)];
R.td_check.rel_err_ideal = [median(errs_id) max(errs_id)];

fid = fopen(fullfile(out, 'gate1.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);
disp(jsonencode(R, 'PrettyPrint', true));

%% helpers
function A = tdl_response(Q, bins, df, rms)
%TDL_RESPONSE Exponential power-delay profile, 12 taps, given RMS delay spread.
%   Stand-in for 3GPP TDL-A/C with the same 30 ns per-hop RMS spread used by
%   Iudice et al.; unit average power.
tau = (0:11) * rms / 2;
pdp = exp(-tau / rms); pdp = pdp / sum(pdp);
A = zeros(Q, numel(bins));
for q = 1:Q
    g = sqrt(pdp/2) .* (randn(1,12) + 1j*randn(1,12));
    A(q, :) = g * exp(-1j*2*pi*tau.' * (bins*df));
end
end

function r = rng_(x)
r = max(x) - min(x);
end
