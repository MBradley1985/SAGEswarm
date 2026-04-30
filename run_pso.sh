#!/bin/bash

# PSO workflow for SAGE — Fig. 26 corner plot run.
# Parameters: α_SF, ε_disk, ε_halo, α_FFB
# Constraints: SMF at z=0, z~6 (snap 18), z~7 (snap 16) — stacked χ²

CONFIG_PATH="../SAGE26/input/millennium.par"
BASE_PATH="../SAGE26/sage"
OUTPUT_PATH="./millennium_pso"
PARTICLES=10
ITERATIONS=5
TEST="chi2"
CONSTRAINTS="SMF_z0"
AGE_ALIST_FILE_MINI_MILLENNIUM="../SAGE26/input/millennium/trees/millennium.a_list"
BOXSIZE=62.5
SIM_MINI_MILLENNIUM=1
VOL_FRAC=1.0
OMEGA0=0.25
H0=0.73
CSVOUTPUT="./millennium_pso/pso.csv"
SPACEFILE="./space.txt"

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
  --sim "$SIM_MINI_MILLENNIUM" \
  --boxsize "$BOXSIZE" \
  --vol-frac "$VOL_FRAC" \
  --Omega0 "$OMEGA0" \
  --h0 "$H0" \
  -S "$SPACEFILE"
