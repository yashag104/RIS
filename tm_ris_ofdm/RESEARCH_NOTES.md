# Time-modulated RIS + OFDM: literature map and candidate paper (2026-10-01)

## Anchor paper
Iudice, Darsena, Gelli, Galdi, "Physically Consistent Modeling of Dispersive
Time-Modulated RIS for Wideband OFDM", arXiv 2609.18360 (16 Sep 2026).

* LPTV model. Each element switches periodically with period T = T_u (the OFDM
  useful symbol), with K slots, and each state has a single-pole dispersive response
  Gamma_k(f) = -1 + 2 xi_r / (j 2 pi (f - f0) + xi).
* Harmonic h couples transmitted subcarrier m to received subcarrier m+h, weighted by
  the element response at the absolute frequency of m. The coupling matrix is
  non-diagonal and, under dispersion, NOT Toeplitz.
* Generalised CP: T_cp >= delay spread + RIS memory T_gamma (13.8 ns for the OpenRIS cell).
* Validation: OpenRIS varactor cell, n78, 100 MHz, 30 kHz SCS. Table I gives fitted
  f0, xi_r, xi_i for the two states.
* Explicitly out of scope / future work: joint equalisation across coupled
  subcarriers, Tx precoding, guard subcarriers, control sequences that suppress
  harmonics, multi-resonant elements, mutual coupling, non-ideal transients,
  oblique incidence, OTA validation.

## Who uses TM-RIS/TMA with OFDM (all assume IDEAL, frequency-flat switching)
* Xu & Petropulu, TM-IRS for waveform security, arXiv 2310.13210 (ICASSP 2024)
* Tao & Petropulu, Secure TM-IRS via GFlowNets, arXiv 2506.14992. The legitimate user
  treats harmonic ICI as interference.
* Tao, Petropulu, Poor, TM-IRS for ISAC + security, arXiv 2509.05565. Future work:
  "hardware impairments".
* Tao, Xu, Petropulu, "How secure is TMA-enabled OFDM DM?", arXiv 2310.08551 (TWC 2025).
  An eavesdropper recovers symbols by blind ICA, resolving ambiguities with the
  TOEPLITZ structure of the mixing matrix (ideal model).
* Verde, Darsena, Galdi, rapidly time-varying RIS, arXiv 2304.11912. No OFDM.
* TMA antenna literature has studied non-ideal switch rise times and element bandwidth
  as a sideband filter (out-of-band), but not in-band OFDM coupling.

## Gate 0 numbers (Iudice Table I, computed here)
OpenRIS cell across 100 MHz: common phase drift 47.3 deg, differential 180 deg +/- small,
imbalance < 0.6 dB, DC-harmonic leakage -19..-27 dB (ideal: -inf). So dispersion is a
second-order effect on the harmonic magnitudes, but it breaks the Toeplitz structure.

## Second novelty sweep (2026-10-01): related areas that must be cited and differentiated
* BEM (basis-expansion model) estimation of doubly selective OFDM channels
  (CE-/P-BEM; e.g. "A BEM for estimation of time-varying channels in OFDM"; RIS-OFDM
  BEM + CNN, Shao et al., Syst. Eng. Electron. 2025). There the time variation is
  *unknown* (mobility/Doppler) and expanded on a chosen basis. Ours: the time
  variation is the *known* RIS modulation (sequence + element physics), so only the
  static two-hop channel is unknown, with rank-1 structure per element.
* PLC linear periodically time-varying channels (mains-synchronous, 10–20 ms period).
  The period is much longer than the OFDM symbol, so there is no harmonic coupling at
  subcarrier spacing. Different regime.
* TMA retrodirective DM with a legitimate-user pilot (Tao/Xu/Petropulu): the pilot is
  used by the *array* to set weights, not by the receiver to estimate the mixing matrix.
* Space-time-coding metasurface sensing/DoA work: harmonic processing for sensing,
  not communication channel estimation.
Result: still no paper on receiver-side estimation/equalisation of the
TM-RIS harmonic-coupled OFDM channel with known modulation.

## Gap (searched; no paper found)
No paper estimates or equalises the harmonic-coupled OFDM channel of a TM-RIS at a
legitimate receiver with pilots, under either the ideal or the dispersive model.
Every TM-RIS paper treats the coupling as interference. The only "undo the mixing" work
is the blind eavesdropper attack above (ideal, Toeplitz, TMA).
