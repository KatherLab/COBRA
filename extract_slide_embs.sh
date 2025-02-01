#!/bin/bash


FEAT_DIR="/path/to/input"
OUTPUT_DIR="/path/to/output"

python -m cobra.inference.extract_feats --feat_dir $FEAT_DIR --output_dir $OUTPUT_DIR