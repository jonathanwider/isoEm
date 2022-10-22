#!/bin/bash

function help {
  echo "Use cdo to interpolate datasets from HadCM3 grid to ico."
  echo "Usage: [-f \"file1,file2,...\"] [-g \"grid\"] [-o \"original_file\"][-i]"
  echo "Options"
  echo "  -f List of HadCM3 files to be interpolated. Files names should end in _5_nbs.nc or 6_nbs.nc. No mixing allowed"
  echo "  -g Grid that is used in all of the specified files"
  echo "  -o Grid of the original file that we want to interpolate to"
  echo "  -i cdo interpolation mode. Valid arguments: NN, cons1"
}

# Defaults
SCRIPT_DIR="."
INTERPOLATION=""

# read options
while getopts ":f:g:o:i:h" OPTION; do
  case $OPTION in
    f) FILES=$OPTARG
       ;;
    g) GRID=$OPTARG
       ;;
    o) ORIGINAL=$OPTARG
       ;;
    i) INTERPOLATION=$OPTARG
       ;;
    h) help
  esac
done
shift $((OPTIND-1))

FILES_ARR=($FILES)
echo $FI
# do the interpolation
for file in $FILES
do
	nbs=$(echo "$GRID"| cut -d '_' -f 6)
	res=$(echo "$GRID"| cut -d '_' -f 4)
	filename=$(echo "$file"| cut -d '.' -f 1)
	cdo -f nc remapcon,"$SCRIPT_DIR/${ORIGINAL}" -setgrid,"$SCRIPT_DIR/${GRID}" "$SCRIPT_DIR/$file" "$SCRIPT_DIR/${filename}_r_${res}_nbs_${nbs}_${INTERPOLATION}.nc"
done
