#!/usr/bin/env bash
set -euo pipefail
trap "echo '腳本被中斷'; exit 130" INT

# 固定資料夾（Set12 原圖)
cfolder="data/Set12"

# 跑 kernel1 ~ kernel8
for k in {1..8}; do
  bfolder="data/Set12_blur/kernel${k}"
  rpath1="results/kernel${k}/blur_noised"
  rpath2="results/kernel${k}/deblur"
  kpath="data/kernels/kernel${k}.png"

  # 收集檔案清單（按自然序排序）
  mapfile -t bfiles < <(ls "$bfolder"/*.png | sort -V)
  mapfile -t cfiles < <(ls "$cfolder"/*.png | sort -V)

  len=${#bfiles[@]}
  if [ "$len" -eq 0 ]; then
    echo "警告：$bfolder 沒有 .png 檔，跳過。"
    continue
  fi
  if [ "$len" -ne "${#cfiles[@]}" ]; then
    echo "兩個資料夾數量不一致！(blur: $len, clean: ${#cfiles[@]}) 於 kernel${k}"
    exit 1
  fi

  # 逐對處理
  for ((i=0; i<len; i++)); do
    python main.py "${bfiles[i]}" "${cfiles[i]}" "${rpath1}" "${rpath2}" "${kpath}"
  done
done