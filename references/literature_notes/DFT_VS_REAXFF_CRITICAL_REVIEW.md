# DFT/QM-MM versus ReaxFF wall-impact conclusions

## Scope

This review compares Laws and Petro, *Journal of Applied Physics* 139,
113302 (2026), with Bendimerad and Petro, *Journal of Electric Propulsion* 1,
27 (2022). The former used 42 DFT/QM-MM trajectories for neutral EMI-BF4 on a
rigid explicit Au slab at 10-100 eV. The latter's reactive Model 2 used 3,120
neutral EMI-BF4 trajectories at 300 K over 10-1000 eV with ReaxFF and a fixed
continuum Lennard-Jones wall.

## Major advances provided by DFT/QM-MM

1. **Electronic charge redistribution and fragment charge state.** DFT/QM-MM
   evaluated the projectile electronically and assigned terminal fragment
   charges from time-resolved RESP charges. It therefore separated neutral,
   cationic, and anionic products. The ReaxFF paper produced pseudomass spectra
   but did not provide an equivalent charge-resolved product spectrum. Its
   authors also state that the charge-equilibration algorithm is ill-defined
   for charged systems (ReaxFF paper, p. 7).

2. **Three physically distinct energy regimes.** DFT identified ionic
   dissociation at 10-20 eV, neutralization at 30-40 eV, and broad covalent
   fragmentation above 50 eV (DFT paper, Fig. 7 and pp. 9-10). The ReaxFF
   study instead reported covalent fragmentation beginning at 10 eV. Thus,
   ReaxFF did not recover the DFT separation between low-energy ionic
   dissociation and higher-energy covalent breakup.

3. **HF and H2 radical recombination.** The DFT paper explicitly states that
   classical approaches without electronic degrees of freedom cannot predict
   radical recombination into neutral HF and H2 (p. 3). DFT identified
   BF4 -> BF3 + HF as a defining 30-40 eV neutralization channel. The published
   ReaxFF spectra label free H, F, BF2, BF3, and BF4 but not HF, and the matched
   42-case ReaxFF rerun contains no exact HF component in any stored frame.

4. **Neutralization products invisible to charged-particle diagnostics.** DFT
   resolved neutral BF3, HF, and a neutralized imidazolium backbone and linked
   these products to residual-gas-analyzer measurements. ReaxFF pseudomass
   alone cannot establish whether a given nominal-mass product is neutral or
   charged.

5. **Energy transfer and product kinematics.** DFT reported fragment-resolved
   kinetic-energy partitioning, scattering angles, and a mass-to-deflection
   anti-correlation. It also identified a maximum in transient metastable
   products near 50 eV. These quantities were not resolved in the ReaxFF
   paper's pseudomass analysis.

6. **Closer representation of the intended Au collision geometry.** DFT used
   an explicit 2,000-atom Au slab, whereas the ReaxFF paper replaced gold with
   an impact-surface-agnostic 12-6 wall because its parameter set contained no
   Au. The ReaxFF parameters had been compared with DFT for an ionic-liquid
   electrolyte on bismuth, not EMI-BF4 impact chemistry on gold (ReaxFF paper,
   pp. 7 and 10).

## What the ReaxFF study failed to identify

| Missing or conflicting result | Evidence |
|---|---|
| HF as a molecular product | No labeled HF in the published ReaxFF EMI-BF4 spectrum; 0/42 exact HF in the matched rerun |
| H2 recombination | Not represented or reported |
| 30-40 eV neutralization window | DFT-specific regime characterized by BF3 + HF and neutralized EMI |
| Charge state of each fragment | ReaxFF reported pseudomass; DFT used terminal RESP-derived charge states |
| Charged products from a neutral projectile | DFT directly resolved mixed charge states; ReaxFF paper did not provide an equivalent analysis |
| Separation of ionic dissociation from covalent fragmentation | ReaxFF reported covalent breakup from 10 eV; DFT placed prevalent covalent breakup above 50 eV |
| Fate of free hydrogen | ReaxFF authors state that their temporal and spatial scales were insufficient to capture it (p. 9) |
| Explicit Au-specific impact response | ReaxFF used a continuum wall because Au parameters were unavailable |
| Fragment energy partition, charge-resolved scattering, and metastable maximum | Not analyzed in the ReaxFF pseudomass study |

## Evidence qualification

DFT/QM-MM has higher electronic and mechanistic fidelity, but the papers do not
constitute a controlled force-field benchmark. They use different walls,
trajectory ensembles, energy ranges, and analysis definitions. DFT sampled
only six orientations per energy and required up to 60 days per trajectory.
Its Au slab was classical and frozen, which suppresses surface phonon energy
transfer; the DFT authors consequently treat large fragment multiplicities and
deflections as upper bounds. ReaxFF sampled far more orientations and is
millions of times faster in the local implementation.

The defensible conclusion is therefore not that every DFT result is proven
correct. It is that DFT/QM-MM resolves charge redistribution, neutralization,
and HF-forming chemistry that the published ReaxFF model either cannot
represent reliably or did not recover, while ReaxFF remains substantially
better suited to broad statistical sampling.
