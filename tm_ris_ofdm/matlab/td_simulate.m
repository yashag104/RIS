function Y = td_simulate(X, bins, Nfft, fc, df, seqs, As, Ad, P)
%TD_SIMULATE Independent time-domain reference for one OFDM symbol.
%   Implements the quasi-static LPTV element directly, with no harmonic algebra:
%   for each element q and slot state k, filter the (cyclically extended)
%   symbol with Gamma_k at absolute frequency, keep it only during the slots
%   where state k is active, and sum. Propagation is applied as per-element
%   frequency responses before (As) and after (Ad) the surface.
%
%   X    : symbols on active bins (numel(bins) x 1); bins are signed indices
%   Nfft : samples per useful symbol (oversample so harmonics do not alias)
%   Returns Y, the demodulated active bins.
%
%   The output-side response Ad must be applied to the switched signal, which
%   now occupies every bin, so Ad is evaluated on the full FFT grid by the
%   caller-provided function handle when Ad is a handle, or zero-padded to the
%   active bins otherwise.

Q = size(seqs, 1);
K = size(seqs, 2);
allbins = [0:Nfft/2-1, -Nfft/2:-1];            % FFT bin order
pos = mod(bins, Nfft) + 1;                     % FFT positions of active bins
n = (0:Nfft-1).';
slot = floor(n*K/Nfft) + 1;                    % slot index per sample
Yf = zeros(Nfft, 1);
for q = 1:Q
    Xq = zeros(Nfft, 1);
    Xq(pos) = X(:) .* As(q, :).';
    z = zeros(Nfft, 1);
    for s = 1:K
        st = seqs(q, s);
        Gk = element_response(fc + allbins.'*df, st, P);
        xk = ifft(Xq .* Gk);                   % steady-state (CP absorbs memory)
        z(slot == s) = z(slot == s) + xk(slot == s);
    end
    Z = fft(z);
    if isa(Ad, 'function_handle')
        Yf = Yf + Z .* Ad(q, allbins.');
    else
        A = zeros(Nfft, 1); A(pos) = Ad(q, :).';
        Yf = Yf + Z .* A;
    end
end
Y = Yf(pos);
end
