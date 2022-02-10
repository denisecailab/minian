## v1.2.1 (2022-02-10)

### Fix

- avoid syntax error in `update_spatial` returns

## v1.2.0 (2022-02-09)

### Feat

- use least square to produce proper scaling in temporal components and background terms

### Fix

- rescale with normalizing factor when using `normalize` parameter in spatial and temporal update
- fix unit id mismatch in spatial parameter exploration

## v1.1.0 (2021-09-10)

### Fix

- pin jinja2 version to avoid doc build fail
- use fft filter for peak-to-noise ratio computation
- avoid conversion in `xrconcat_recursive`

### Feat

- baseline fluorescence correction in temporal update with median filter

## v1.0.1 (2021-05-05)

### Fix

- fix various typo and improve instructions in notebook

## v1.0.0 (2021-05-03)

### Highlight

- use dask localcluster and throttling for all computations to reduce memory demands
- add dedicated documentation site
- add testing and continuous integration
- release on conda-forge

## v1.0.0rc1 (2021-04-30)

### Feat

- graph based resolving of mappings

### Fix

- fix pipeline when `subset` is used

## v1.0.0rc0 (2021-04-11)

Candidate for first public release.
