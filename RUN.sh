#!/usr/bin/env bash
set -euo pipefail
trap "echo 'Script interrupted'; exit 130" INT

# Fixed folder (Set12 clean images)
cfolder="data/sharp"

# Run kernel1 ~ kernel8
for k in {1..8}; do
  bfolder="data/blur/kernel${k}"
  rpath1="results/kernel${k}/blur_noised"
  rpath2="results/kernel${k}/deblur"
  kpath="data/kernels/kernel${k}.png"

  # Collect file lists (natural sort)
  mapfile -t bfiles < <(ls "$bfolder"/*.png | sort -V)
  mapfile -t cfiles < <(ls "$cfolder"/*.png | sort -V)

  len=${#bfiles[@]}
  if [ "$len" -eq 0 ]; then
    echo "Warning: No .png files found in $bfolder. Skipping."
    continue
  fi
  if [ "$len" -ne "${#cfiles[@]}" ]; then
    echo "Mismatch in number of files! (blur: $len, clean: ${#cfiles[@]}) at kernel${k}"
    exit 1
  fi

  # Process each image pair
  for ((i=0; i<len; i++)); do
    python main.py "${bfiles[i]}" "${cfiles[i]}" "${rpath1}" "${rpath2}" "${kpath}"
  done
done