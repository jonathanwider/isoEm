#!/bin/bash

function help {
  echo "Use cdo to interpolate datasets from HadCM3 grid to ico."
  echo "Usage: [-f \"file1,file2,...\"] [-g \"grid6nb,grid5nb\"] [-i]"
  echo "Options"
  echo "  -f List of HadCM3 files to be interpolated. The interpolation assumes that all variables live on the horizontal grid of the first given file."
  echo "  -g List of grid description filenames"
  echo "  -i cdo interpolation mode. Valid arguments: NN, cons1"
}

# Defaults
SCRIPT_DIR="."
INTERPOLATION=""

# read options
while getopts ":f:g:i:h" OPTION; do
  case $OPTION in
    f) FILES=$OPTARG
       ;;
    g) GRIDS=$OPTARG 
       ;;
    i) INTERPOLATION=$OPTARG 
       ;;
    h) help
  esac
done
shift $((OPTIND-1))


FILES_ARR=($FILES)
echo $FI

# generate weights
if [ "$INTERPOLATION" == "cons1" ]; then
	for grid in $GRIDS
	do
	  nbs=$(echo "$grid"| cut -d '_' -f 6)
	  res=$(echo "$grid"| cut -d '_' -f 4)
	  cdo gencon,"${SCRIPT_DIR}/${grid}" "$SCRIPT_DIR/${FILES_ARR[0]}" "$SCRIPT_DIR/weights_${INTERPOLATION}_nbs_${nbs}_res_${res}.nc"
	done
elif [ "$INTERPOLATION" == "NN" ]; then
	for grid in $GRIDS
	do
	  nbs=$(echo "$grid"| cut -d '_' -f 6)
	  res=$(echo "$grid"| cut -d '_' -f 4)
	  cdo gennn,"${SCRIPT_DIR}/${grid}" "$SCRIPT_DIR/${FILES_ARR[0]}" "$SCRIPT_DIR/weights_${INTERPOLATION}_nbs_${nbs}_res_${res}.nc"
	done
else  
	echo "Invalid interpolation method!"
	help
fi

echo "done"
# do the interpolation
for file in $FILES
do 
	for grid in $GRIDS
	do
	  nbs=$(echo "$grid"| cut -d '_' -f 6)
	  res=$(echo "$grid"| cut -d '_' -f 4)
	  filename=$(echo "$file"| cut -d '.' -f 1)
	  cdo -f nc remap,"$SCRIPT_DIR/$grid","$SCRIPT_DIR/weights_${INTERPOLATION}_nbs_${nbs}_res_${res}.nc" "$SCRIPT_DIR/$file" "$SCRIPT_DIR/${filename}_r_${res}_nbs_${nbs}_${INTERPOLATION}.nc"
	done
done
