#!/bin/bash


# This batch script runs the PSO workflow for SAGE.
# If any required sage_*.csv files are missing in ./data/, they will be automatically generated from the .hdf5 output after SAGE runs.
# The output directory is parsed from the OutputDir line in the .par file.

CONFIG_PATH="../../SAGE-2.0/sage-model/input/millennium.par"
BASE_PATH="../../SAGE-2.0/sage-model/sage"
OUTPUT_PATH="../../SAGE-2.0/sage-model/output/millennium_pso"
PARTICLES=16
ITERATIONS=10
TEST="chi2"
CONSTRAINTS="SMF_z0,BHMF_z0,BHBM,CSFRDH"
AGE_ALIST_FILE_MINI_UCHUU='/fred/oz004/msinha/simulations/uchuu_suite/miniuchuu/mergertrees/u400_planck2016_50.a_list'
AGE_ALIST_FILE_MINI_MILLENNIUM="../../SAGE-2.0/sage-model/input/millennium/trees/millennium.a_list"
BOXSIZE=62.5
SIM_MINI_UCHUU=0
SIM_MINI_MILLENNIUM=1
VOL_FRAC=1.0
OMEGA0=0.25 
H0=0.73
CSVOUTPUT="../../SAGE-2.0/sage-model/output/millennium_pso/params_z0.csv"
SPACEFILE="./space.txt"
ACCOUNT="oz004"

python3 ./main.py \
  -c "$CONFIG_PATH" \
  -b "$BASE_PATH" \
  -o "$OUTPUT_PATH" \
  -s "$PARTICLES" \
  -m "$ITERATIONS" \
  -t "$TEST" \
  -x "$CONSTRAINTS" \
  -csv "$CSVOUTPUT" \
  --age-alist-file "$AGE_ALIST_FILE_MINI_MILLENNIUM" \
  --sim "$SIM_MINI_MILLENNIUM"\
  --boxsize "$BOXSIZE" \
  --vol-frac "$VOL_FRAC" \
  --Omega0 "$OMEGA0" \
  --h0 "$H0" \
  -S "$SPACEFILE"
