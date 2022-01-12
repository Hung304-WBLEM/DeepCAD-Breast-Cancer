- process_cbis_ddsm.py: split train val, crop lesions from mammograms, create mass_shape, mass_margins, calc_type, calc_dist dataset

- check_cbis_ddsm.py: check if raw data cbis_ddsm contain enough samples (by using the provided csv files)
- reorganize_cbis_ddsm.py: reorganze cbis_ddsm dataset using the provided csv files.
the original dataset downloaded from the Cancer Imaging Archive is not organize in the correct files structure
- load_cbis_ddsm.py: using tensorflow dataset to extract augmented patches from mammograms
