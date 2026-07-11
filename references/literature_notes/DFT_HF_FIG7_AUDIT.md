# DFT HF audit from Laws and Petro Figure 7

## Directly supported

Figure 7 shows a terminal mass-20 HF bar at every simulated energy. The
digitized probabilities are 0.08075, 0.05590, 0.04658, 0.14596, 0.02950,
0.10559, and 0.10093 at 10, 20, 30, 40, 50, 75, and 100 eV, respectively.
The paper identifies BF4 -> BF3 + HF as a defining neutralization pathway at
30-40 eV and assigns terminal fragment charge states at 2 ps.

## Integer-count reconstruction

The visible bar probabilities nearly sum to one and recur at integer multiples
of a common energy-specific increment. Fitting the digitized bars to `k/N`
gives likely total terminal-fragment denominators of 12, 17, 20, 27, 30, 38,
and 49. Multiplying the HF probability by these denominators gives likely HF
fragment counts of 1, 1, 1, 4, 1, 4, and 5.

This reconstruction is strongest at 10-40 eV, where all fitted fragment counts
are represented by extracted bars. At 50 eV, the automatic color mask recovered
only about 80% of the inferred fragment population, so that count has low
confidence. The 75 and 100 eV fits are nearly complete but remain inferred.

## Why these are not trajectory counts

Each spectrum pools terminal fragments from six molecular orientations. Figure
7 does not map a fragment to its parent trajectory, and one EMI-BF4 projectile
contains four fluorine atoms, so a trajectory can in principle generate more
than one HF molecule. Consequently, one inferred HF fragment implies exactly
one HF-positive trajectory, four HF fragments imply between one and four
positive trajectories, and five imply between two and five.

The exact DFT trajectory-level total is therefore bounded by 8-17 positive
cases out of 42 under the inferred fragment counts, not established as a single
number. Assuming at most one HF molecule per trajectory would give 17/42, but
that assumption is not stated in the paper and must not be reported as an
observed DFT result.

## Comparison implication

MACE-medium has directly observed HF in 11/42 stored trajectories and
MACE-POLAR-1 in 19/42. These values overlap or bracket the DFT figure-derived
8-17/42 bound, but the comparison is not quantitative because the DFT bound is
reverse-engineered from pooled terminal spectra and the MACE protocols use
different surfaces and trajectory storage. ReaxFF has directly observed HF in
0/42 stored trajectories, which is qualitatively inconsistent with the clear
DFT HF channel.
