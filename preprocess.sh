#!/bin/bash

function help {
  echo "Use cdo to preprocess dataset. First the valid range of the isotope dataset is set to (-100,100), then yearly averages are computed and stored in new files"
  echo "Usage: [-f \"file1,file2,...\"], [-dmin \"d18O_min\"] [-dmax \"d18O_max\"]"
  echo "Options"
  echo "  -f Name of the file to be preprocessed."
}

# Defaults
SCRIPT_DIR="."
DMIN=-100
DMAX=100

# read options
while getopts ":f:g:o:i:h" OPTION; do
  case $OPTION in
    f) FILES=$OPTARG
       ;;
    dmin) DMIN=$OPTARG
       ;;
    dmax) DMAX=$OPTARG
       ;;
    h) help
  esac
done
shift $((OPTIND-1))

FILES_ARR=($FILES)

echo $FI
# do the interpolation
for file in $FILES; do
  if [[ "$file" == *"isotopes"* ]];then
    # if the file name contains isotopes then
    # current default behaviour: Overwrite existing file.
    cdo setvrange,$DMIN,$DMAX file file
  elif [[ "$file" == *"tsurf"* ]];then
    :  # can be implemented later if needed
  elif [[ "$file" == *"prec"* ]];then
    :  # can be implemented later if needed
  else
    echo "Invalid filename! Filename must contain one element of [isotopes, tsurf, prec]." 1>&2
    exit 1
  fi
done
