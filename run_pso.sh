#!/bin/bash

# Box size, cosmology, processed volume fraction, the scale-factor list and
# the snapshot-to-redshift mapping are all read from the SAGE output header,
# so there is nothing here to keep in sync with the simulation. Switching
# simulation is just CONFIG_PATH.

CONFIG_PATH="../SAGE26/input/millennium.par"
BASE_PATH="../SAGE26/sage"
OUTPUT_PATH="./millennium_pso"
PARTICLES=10
ITERATIONS=50
TEST="chi2"
CONSTRAINTS="SMF_z0,CSFRDH"
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
  -S "$SPACEFILE"
