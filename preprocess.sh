#!/bin/bash

function help {
  echo "Use cdo to preprocess dataset. First the valid range of the isotope dataset is set to (-100,100), then yearly averages are computed and stored in new files"
  echo "Usage: [-f \"file1,file2,...\"], [-g \"grid\"][-l \"d18O_min\"] [-u \"d18O_max\"]"
  echo "Options"
  echo "  -f Name of the file to be preprocessed."
  echo "  -g Grid to be interpolated to. Can also be a netCDF4 file."
  echo "  -l Lower bound of the isotope range. Default: -100."
  echo "  -u Upper bound of the isotope range. Default: 100."
  echo "  -h Display help"
}

# Defaults
DMIN=-100
DMAX=100

# read options
while getopts ":f:l:u:g:h" OPTION; do
  case $OPTION in
    f) FILES=$OPTARG
       ;;
    l) DMIN=$OPTARG
       ;;
    u) DMAX=$OPTARG
       ;;
    g) GRID=$OPTARG
       ;;
    h) help
  esac
done
shift $((OPTIND-1))

FILES_ARR=($FILES)

# do the preprocessing
for file in $FILES; do
  filename="${file%%_*}" # get rid of file ending
  filename="${file%%_*}" # get rid of file ending

  if [[ "$file" == *"isotopes_raw"* ]];then
    # if the file name contains isotopes then
    # current default behaviour: Overwrite existing files
    cdo setvrange,$DMIN,$DMAX "${filename}_raw.nc" "${filename}_tmp.nc"
    if [[ "$file" != *"iHadCM3"* ]];then
      cdo remapbil,$GRID "${filename}_tmp.nc" "${filename}.nc"
    else
      cdo copy "${filename}_tmp.nc" "${filename}.nc"
    fi
    cdo yearmean "${filename}.nc" "${filename}_yearly.nc"
  elif [[ "$file" == *"tsurf_raw"* ]];then
    cdo setvrange,173,343 "${filename}_raw.nc" "${filename}_tmp.nc" # set valid precipitation value range to -100 to 70 °C
    if [[ "$file" != *"iHadCM3"* ]];then
      cdo remapbil,$GRID "${filename}_tmp.nc" "${filename}.nc"
    else
      cdo copy "${filename}_tmp.nc" "${filename}.nc"
    fi
    cdo yearmean "${filename}.nc" "${filename}_yearly.nc"
  elif [[ "$file" == *"prec_raw"* ]];then
    cdo setvrange,-1,10000 "${filename}_raw.nc" "${filename}_tmp.nc" # set valid precipitation value range to -1 to 10000 mm/month
    if [[ "$file" != *"iHadCM3"* ]];then
      cdo remapbil,$GRID "${filename}_tmp.nc" "${filename}.nc"
    else
      cdo copy "${filename}_tmp.nc" "${filename}.nc"
    fi
    cdo yearmean "${filename}.nc" "${filename}_yearly.nc"
  else
    echo "Invalid filename! Filename must contain one element of [isotopes, tsurf, prec]." 1>&2
    exit 1
  fi
  find -type f -name '*tmp*' -delete
done
