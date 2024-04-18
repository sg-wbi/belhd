#!/usr/bin/env bash

EXP_DIR="/vol/fob-wbia-vol1/wbi/gardasam/data/belb-dr/nohd"

# python train_nohd.py exp_dir="$EXP_DIR" train=gnormplus.train.gene kb_subset=gnormplus global_candidates=true kb_subset=gnormplus run_id=gnormplus 
# python train_nohd.py exp_dir="$EXP_DIR" train=gnormplus.train.gene kb_subset=gnormplus global_candidates=true kb_subset=gnormplus abbres=true run_id=gnormplus_ar 

# python train_nohd.py exp_dir="$EXP_DIR" train=nlm_gene.train.gene kb_subset=nlm_gene global_candidates=true kb_subset=nlm_gene run_id=nlm_gene
# python train_nohd.py exp_dir="$EXP_DIR" test=nlm_gene.train.gene kb_subset=nlm_gene global_candidates=true kb_subset=nlm_gene abbres=true run_id=nlm_gene_ar 

# accelerate launch --multi_gpu --num_processes=2 train_nohd.py exp_dir="$EXP_DIR" train=nlm_gene.train.gene kb_subset=nlm_gene global_candidates=true kb_subset=nlm_gene run_id=nlm_gene
accelerate launch --multi_gpu --num_processes=2 train_nohd.py exp_dir="$EXP_DIR" train=nlm_gene.train.gene kb_subset=nlm_gene global_candidates=true kb_subset=nlm_gene abbres=true run_id=nlm_gene_ar

# python train_nohd.py exp_dir="$EXP_DIR" train=medmentions.train.umls global_candidates=true run_id=medmentions 
# python train_nohd.py exp_dir="$EXP_DIR" train=medmentions.train.umls global_candidates=true abbres=true run_id=medmentions_ar 
