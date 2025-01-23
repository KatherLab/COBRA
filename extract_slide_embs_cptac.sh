#!/bin/bash

# Parse command-line arguments
# Example usage: ./extract_slide_embsII.sh -j <job_id> -c <cohort1,cohort2,...>
# i.e.: ./extract_slide_embsII.sh -j 12345 -c cohort1,cohort2,cohort3

while getopts j:c:m:d: flag
do
    case "${flag}" in
        j) job_id=${OPTARG};;
        c) cohorts=${OPTARG};;
        m) model_dir=${OPTARG};;
        d) config_dir=${OPTARG};;
    esac
done

CONFIG_DIR="${config_dir}"
CHECKPOINT_DIR="${model_dir}"
if [[ "$CONFIG_DIR" != *"$job_id"* ]]; then
       echo "Error: job_id not found in config_dir"
       exit 1
fi

if [[ "$CHECKPOINT_DIR" != *"$job_id"* ]]; then
       echo "Error: job_id not found in checkpoint_dir"
       exit 1
fi
IFS=',' read -r -a cohort_array <<< "$cohorts"

for cohort in "${cohort_array[@]}"
do
    OUTPUT_DIR="/data/cat/ws/s1787956-cobra/data/cobraII-feats-cptac-${job_id}/${cohort}"
    FEAT_DIR="/data/cat/ws/s1787956-cobra/data/features-cptac/features-20x/virchow2/CPTAC-${cohort}/virchow2-stamp-maru-21-12-24"
    SLIDE_TABLE="/data/cat/ws/s1787956-cobra/slide_tables/slide_table_cptac_${cohort,,}.csv"
    
    python -m cobra.inference.extract_feats_patient -f $FEAT_DIR -o $OUTPUT_DIR -d -c $CONFIG_DIR -w $CHECKPOINT_DIR -m "cobraII-${job_id}" -e "cobraII_feats_${cohort}.h5" -s $SLIDE_TABLE
done
