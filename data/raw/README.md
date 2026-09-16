# Palmer Penguins source data

`penguins.csv` is the unmodified, simplified Palmer Penguins dataset supplied by
the palmerpenguins project: 344 records, eight columns, three species.

- Download: https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv
- Dataset documentation: https://allisonhorst.github.io/palmerpenguins/
- License: CC0 (https://creativecommons.org/publicdomain/zero/1.0/)
- Retrieved: 2026-09-16
- SHA-256: `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`

Citation: Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago
(Antarctica) penguin data*. https://doi.org/10.5281/zenodo.3960218

Data were collected and made available by Dr. Kristen Gorman and Palmer Station
Antarctica LTER. Original research: Gorman KB, Williams TD, Fraser WR (2014).
*Ecological sexual dimorphism and environmental variability within a community
of Antarctic penguins (genus Pygoscelis)*. PLoS ONE 9(3):e90081.
https://doi.org/10.1371/journal.pone.0090081

The pipeline reads this committed local file; it does not download data on each
run. It uses the four physical measurements and species, ignoring sex, island,
and year. Missing values are preserved in the source so the preparation stage
can handle them explicitly. This dataset is neither CelebFaces nor the patient's
smoking status dataset excluded by the assignment.
