#!/bin/bash

function help {
  echo "Use cdo to preprocess dataset. First the valid range of the isotope dataset is set to (-100,100), then yearly averages are computed and stored in new files"
  echo "Usage: [-f \"file1,file2,...\"], [-l \"d18O_min\"] [-u \"d18O_max\"]"
  echo "Options"
  echo "  -f Name of the file to be preprocessed."
  echo "  -l Lower bound of the isotope range. Default: -100."
  echo "  -u Upper bound of the isotope range. Default: 100."
  echo "  -h Display help"
}

# Defaults
DMIN=-100
DMAX=100

# read options
while getopts ":f:l:u:h" OPTION; do
  case $OPTION in
    f) FILES=$OPTARG
       ;;
    l) DMIN=$OPTARG
       ;;
    u) DMAX=$OPTARG
       ;;
    h) help
  esac
done
shift $((OPTIND-1))

FILES_ARR=($FILES)

echo $DMIN
echo $DMAX
echo $FI
# do the interpolation
for file in $FILES; do
  filename="${file%.*}" # get rid of file ending
  if [[ "$file" == *"isotopes"* ]];then
    # if the file name contains isotopes then
    # current default behaviour: Overwrite existing file.
    cdo setvrange,$DMIN,$DMAX "$file" "${filename}_vrg.nc"
    cdo yearmean "${filename}_vrg.nc" "${filename}_yearly.nc"
  elif [[ "$file" == *"tsurf"* ]];then
    cdo yearmean "${file}" "${filename}_yearly.nc"
  elif [[ "$file" == *"prec"* ]];then
    cdo yearmean "${file}" "${filename}_yearly.nc"
  else
    echo "Invalid filename! Filename must contain one element of [isotopes, tsurf, prec]." 1>&2
    exit 1
  fi
done
