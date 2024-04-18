#!/usr/bin/env bash

EXP_DIR="/vol/fob-wbia-vol1/wbi/gardasam/data/belb-dr/nohd"


# python predict_nohd.py exp_dir="$EXP_DIR" run_id=gnormplus test=gnormplus.test.gene kb_subset=gnormplus
# python predict_nohd.py exp_dir="$EXP_DIR" run_id=gnormplus_ar test=gnormplus.test.gene kb_subset=gnormplus abbres=true

python predict_nohd.py exp_dir="$EXP_DIR" run_id=nlm_gene test=nlm_gene.test.gene kb_subset=nlm_gene
python predict_nohd.py exp_dir="$EXP_DIR" run_id=nlm_gene_ar test=nlm_gene.test.gene kb_subset=nlm_gene abbres=true

# python predict_nohd.py exp_dir="$EXP_DIR" run_id=medmentions test=medmentions.test.umls max_mentions=20
# python predict_nohd.py exp_dir="$EXP_DIR" run_id=medmentions_ar test=medmentions.test.umls max_mentions=20 abbres=true




