## Code for the Nature Communications article "The role of cyclonic activity in tropical temperature-rainfall scaling"

This repository contains the code to reproduce the results and figures of the following article published in Nature Communications:

Traxl, D., Boers, N., Rheinwalt, A., Bookhagen, B., <b>The role of cyclonic activity in tropical temperature-rainfall scaling</b>, Nature Communications, 2021.

### DOI

https:...

### Downloading Original Data

All data necessary to run the scripts in this repository are publicly available. For rainfall estimates, we used the Tropical Rainfall Measuring Mission (TRMM) 3B42 V7 dataset, available trough  
https://disc.gsfc.nasa.gov/datasets/TRMM_3B42_7/summary  
and downloaded from  
https://disc2.gesdisc.eosdis.nasa.gov/s4pa/TRMM_L3/TRMM_3B42.7/  
For temperature estimates, we used the ERA5 reanalysis dataset, downloaded from  
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels  
For sea surface temperatures, we used the NOAA OI SST V2 High Resolution Dataset, available through  
https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html  
and downloaded from  
ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2.highres/  
For tropical cyclone tracks, we used the International Best Track Archive for Climate Stewardship (IBTRACS), available through  
https://www.ncdc.noaa.gov/ibtracs/  
and downloaded from  
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc


### Data Processing Scripts 

The scripts to process the data are located in the `data_processing` subfolder, and have to be run in the indicated order. 

### Figure Scripts

To reproduce the figures of the article, run
  - fig1a_gl_sxt_vs_r_fits.py
  - fig1b_SXT_mean_of_all_time_map.py
  - fig2_fig4_glcp_burst_analysis.py
  - fig3a_gl_grad_stats_mean.py
  - fig3bc_gl_grad_vs_r_fits.py

Note that all scripts in the `data_processing` subfolder need to be executed before running the scripts to create the figures.

### Contact Information

Dominik Traxl (dominik.traxl@posteo.org)

