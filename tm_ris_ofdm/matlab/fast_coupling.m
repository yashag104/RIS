function op = fast_coupling(bins, fc, df, seqs, As, Ad, P)
%FAST_COUPLING Exact O(Q M log M) products with the TM-RIS coupling matrix.
%   For element q, B_q(mbar, m) = b_q^[mbar-m](f_m) = sum_s T_{q,s}(mbar-m) Gamma_s(f_m),
%   where T_{q,s}(h) = exp(-j pi h/K)/K sinc(h/K) sum_{k: seq_q(k)=s} exp(-j 2 pi h k/K)
%   is TOEPLITZ (depends on h only) and Gamma_s is the dispersive response of
%   state s at the input frequency. Hence
%       H x  = sum_q Ad_q .* sum_s T_{q,s} * (Gamma_s .* As_q .* x)
%       H'y  = sum_q conj(As_q) .* sum_s conj(Gamma_s) .* T_{q,s}' * (conj(Ad_q) .* y)
%   with each Toeplitz product done by FFT. Exact on the active band
%   (coupling to inactive bins is dropped, as in COUPLING_MATRIX).
%   Returns op.H(x), op.Ht(y) for x, y of size M x n.

M = numel(bins); Q = size(seqs, 1); K = size(seqs, 2);
f = fc + bins(:)*df;
Gam = [element_response(f, 1, P), element_response(f, 2, P)];      % M x 2
h = (-(M-1):(M-1)).';
env = exp(-1j*pi*h/K)/K .* snc(h/K);
Nf = 2^nextpow2(4*M);
Tf = zeros(Nf, Q, 2); Tfh = Tf;                                     % FFTs of Toeplitz kernels
for q = 1:Q
    for s = 1:2
        ks = find(seqs(q, :) == s) - 1;
        t = env .* sum(exp(-1j*2*pi*h*ks/K), 2);                    % t(h), h = -(M-1)..M-1
        Tf(:, q, s) = fft(t, Nf);
        Tfh(:, q, s) = fft(conj(flipud(t)), Nf);                     % kernel of T'
    end
end
op.H = @(x) apply(x, false);
op.Ht = @(y) apply(y, true);

    function out = apply(x, adj)
        n = size(x, 2); out = zeros(M, n);
        for q = 1:Q
            if ~adj
                acc = zeros(M, n);
                for s = 1:2
                    acc = acc + toep(Tf(:, q, s), Gam(:, s) .* As(q, :).' .* x);
                end
                out = out + Ad(q, :).' .* acc;
            else
                z = conj(Ad(q, :).') .* x; acc = zeros(M, n);
                for s = 1:2
                    acc = acc + conj(Gam(:, s)) .* toep(Tfh(:, q, s), z);
                end
                out = out + conj(As(q, :).') .* acc;
            end
        end
    end

    function y = toep(Tk, x)
        % y(mbar) = sum_m t(mbar - m) x(m), mbar, m in 1..M (linear convolution, central part)
        X = fft(x, Nf);
        c = ifft(Tk .* X);
        y = c(M:(2*M-1), :);
    end
end

function y = snc(x)
y = ones(size(x)); nz = x ~= 0; y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end
