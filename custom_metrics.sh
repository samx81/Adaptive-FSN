#!/usr/bin/env bash
exp_name=$1
ref_dir=$2
est_dir=$3

mkdir -p metrics/$exp_name

python speech_enhance/tools/calculate_metrics.py \
  -R $ref_dir \
  -E $est_dir \
  -M SI_SDR,STOI,WB_PESQ,CSIG,COVL \
  -D metrics/$exp_name/ \
#  -S DNS_1
