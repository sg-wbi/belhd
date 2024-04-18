#!/usr/bin/env bash

EXP_DIR="/vol/fob-wbia-vol1/wbi/gardasam/data/belb-dr/belhd"

python train.py train=ncbi_disease.train.disease global_candidates=true exp_dir="$EXP_DIR" run_id=ncbi_disease
python train.py train=bc5cdr.train.disease global_candidates=true exp_dir="$EXP_DIR" run_id=bc5cdr_disease
python train.py train=bc5cdr.train.chemical global_candidates=true exp_dir="$EXP_DIR" run_id=bc5cdr_chemical
python train.py train=nlm_chem.train.chemical global_candidates=true exp_dir="$EXP_DIR" run_id=nlm_chem
python train.py train=gnormplus.train.gene global_candidates=true exp_dir="$EXP_DIR" run_id=gnormplus kb_subset=gnormplus
python train.py train=nlm_gene.train.gene global_candidates=true exp_dir="$EXP_DIR" run_id=nlm_gene   kb_subset=nlm_gene
python train.py train=linnaeus.train.species global_candidates=true exp_dir="$EXP_DIR" run_id=linnaeus
python train.py train=s800.train.species global_candidates=true exp_dir="$EXP_DIR" run_id=s800
python train.py train=medmentions.train.umls global_candidates=true exp_dir="$EXP_DIR" run_id=medmentions max_mentions=20 refresh_index_every_n=1000

# accelerate launch --multi_gpu --num_processes=4 train.py train=nlm_gene.train.gene global_candidates=true exp_dir="$EXP_DIR" run_id=nlm_gene

