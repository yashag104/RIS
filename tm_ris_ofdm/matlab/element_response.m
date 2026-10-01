function G = element_response(f, k, P)
%ELEMENT_RESPONSE Reflection coefficient of state k at absolute frequency f.
%   Gamma_k(f) = -1 + 2*xi_r / (j*2*pi*(f - f0) + xi_r + xi_i)
%   (single-pole model, Iudice et al. eq. for Gamma_q^[k](f)).
%   P = struct with f0, xr, xi (see PARAMS_OPENRIS), or P = 'ideal' for the
%   frequency-flat binary model used by the TM-IRS literature (+1 / -1).

if ischar(P) && strcmp(P, 'ideal')
    vals = [1, -1];
    G = vals(k) * ones(size(f));
    return
end
G = -1 + 2*P.xr(k) ./ (1j*2*pi*(f - P.f0(k)) + P.xr(k) + P.xi(k));
end
