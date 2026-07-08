"""
Causal-inference layer for the PISA-Albania project.

The predictive work (Phases 1-9) is associational. This package upgrades a
slice of it to a *causal* claim by exploiting the 26 November 2019 M6.4 Durres
earthquake as a natural experiment: PISA's sampling STRATUM encodes a
North / Center / South geographic band, and the quake damage was concentrated
in the central coastal band (Durres + Tirana counties). A difference-in-
differences design compares the change in low-proficiency rates in the
Center band (treated) against the North+South bands (control) from the
pre-quake 2018 cycle to the post-quake 2022 cycle.

Modules:
    region  - decode the STRATUM label into (urbanicity, region band, sector)
    did     - weighted difference-in-differences with PISA design-based SEs
"""
