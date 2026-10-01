function P = params_openris()
%PARAMS_OPENRIS Single-pole fit of the OpenRIS varactor cell, two bias states.
%   Values from Iudice, Darsena, Gelli, Galdi, arXiv 2609.18360, Table I
%   (n78, SMV1408 varactor). Rates are given there as xi/(2*pi) in Hz.
%
%   P.f0(k), P.xr(k), P.xi(k) for state k = 1 (c0, 4 V) and 2 (c1, 19.25 V);
%   xr, xi returned in rad/s.

P.name = 'OpenRIS n78 (Iudice et al., Table I)';
P.f0 = [3.471e9, 3.710e9];
P.xr = 2*pi*[97.5e6, 128.4e6];   % radiative decay rate
P.xi = 2*pi*[8.9e6,  9.0e6];     % intrinsic (loss) decay rate
end
