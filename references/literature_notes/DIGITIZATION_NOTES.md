# Literature mass-spectrum digitization

## DFT-MD

The seven panels in Laws and Petro Fig. 7 were calibrated independently from
the labeled axes. The x calibration uses the 0 and 200 amu ticks, and each
y calibration uses the 0 and 0.250 probability grid lines. A saturation mask
captures the complete charge colormap rather than selecting only one bar color.

The extraction contains 89 visible bars:

| Energy (eV) | Bars |
|---:|---:|
| 10 | 7 |
| 20 | 10 |
| 30 | 9 |
| 40 | 13 |
| 50 | 14 |
| 75 | 16 |
| 100 | 20 |

Estimated raster uncertainties are +/-0.5 amu and +/-0.005 probability. These
values are suitable for visual and conclusion-level comparison, not for
replacing author-supplied numerical data.

## ReaxFF

ReaxFF Fig. 8 overlays ten energy series in one low-resolution raster. The
bundled `scientific-plot-digitizer` color extractor was run separately for each
legend color. Only five segments at 10, 20, 30, and 1000 eV remain visibly
identifiable and pass overlay inspection.

Fully covered bars cannot be reconstructed from the raster. Missing ReaxFF
values therefore mean **occluded or unresolved**, not zero probability. The
50 eV gray-color extraction was rejected because it selected labels and grid
lines. A complete energy-resolved ReaxFF spectrum requires the authors' raw
data or separate, non-overlaid source panels.

## Files

- `dft_mass_spectrum_digitized.csv`: all DFT visible bars.
- `reaxff_visible_peaks_digitized.csv`: QA-passing ReaxFF segments only.
- `literature_mass_spectra_digitized.csv`: combined long-format table.
- `literature_mass_spectra_digitization_metadata.json`: calibration and mask parameters.
- `dft_mass_spectrum_digitization_overlay.png`: DFT QA overlay.
- `reaxff_overlays/`: per-color ReaxFF QA overlays.

Reproduce the DFT extraction and consolidated outputs with:

```bash
python scripts/digitize_literature_mass_spectra.py
```
