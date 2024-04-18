#!/usr/bin/env bash

EXP_DIR="/vol/fob-wbia-vol1/wbi/gardasam/data/belb-dr/belbert"

python predict.py test=ncbi_disease.test.disease  exp_dir="$EXP_DIR" run_id=ncbi_disease
# python predict.py test=bc5cdr.test.disease  exp_dir="$EXP_DIR" run_id=bc5cdr_disease
# python predict.py test=bc5cdr.test.chemical  exp_dir="$EXP_DIR" run_id=bc5cdr_chemical
# python predict.py test=nlm_chem.test.chemical  exp_dir="$EXP_DIR" run_id=nlm_chem
# python predict.py test=bioid.test.cell_line  exp_dir="$EXP_DIR" run_id=nlm_chem
# python predict.py test=gnormplus.test.gene  exp_dir="$EXP_DIR" run_id=gnormplus kb_subset=gnormplus
# python predict.py test=nlm_gene.test.gene  exp_dir="$EXP_DIR" run_id=nlm_gene kb_subset=nlm_gene
# python predict.py test=s800.test.species  exp_dir="$EXP_DIR" run_id=s800
# python predict.py test=linnaeus.test.species  exp_dir="$EXP_DIR" run_id=linnaeus
# python predict.py test=medmentions.test.umls  exp_dir="$EXP_DIR" run_id=medmentions
#
