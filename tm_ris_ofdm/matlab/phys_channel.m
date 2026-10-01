function [As, Ad] = phys_channel(Q, bins, df, fc, opts)
%PHYS_CHANNEL Physically consistent per-element channels for a compact RIS.
%   All elements see the SAME propagation paths (delays, gains); they differ
%   only by the geometric phase exp(-j k u_l . r_q) of each path at element q.
%   Elements on a sqrt(Q) x sqrt(Q) grid at lambda/2 (narrowband array: the
%   aperture delay, < 1 ns, is negligible across the band).
%   Per hop: one LoS path + opts.P NLoS paths, Rician factor opts.K_db,
%   exponential power-delay profile with RMS opts.rms (continuous delays).

lam = 3e8 / fc; k = 2*pi/lam;
n = round(sqrt(Q));
[gx, gz] = meshgrid(0:n-1, 0:n-1);
r = [gx(:), zeros(Q,1), gz(:)] * lam/2;              % Q x 3, surface in x-z plane
As = hop(); Ad = hop();

    function A = hop()
        P = opts.P; K = 10^(opts.K_db/10);
        tau = [0, sort(-opts.rms*log(rand(1, P)))];
        pw = [K/(K+1), (1/(K+1)) * exp(-tau(2:end)/opts.rms) / sum(exp(-tau(2:end)/opts.rms))];
        g = sqrt(pw) .* exp(1j*2*pi*rand(1, P+1));
        u = randn(P+1, 3); u(:,2) = abs(u(:,2)); u = u ./ vecnorm(u, 2, 2);
        ph = exp(-1j*k*(r*u.'));                         % Q x (P+1)
        A = (ph .* g) * exp(-1j*2*pi*tau.' * (bins(:).'*df));
    end
end
