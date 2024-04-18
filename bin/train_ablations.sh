#!/usr/bin/env bash
#
EXP_DIR="/vol/fob-wbia-vol1/wbi/gardasam/data/belb-dr/belhd_ablations"

# # no global candidates
# python train.py train=nlm_gene.train.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene run_id=nlm_gene_nocs
# # no projection layer
# python train.py train=nlm_gene.train.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene global_candidates=true project=-1 run_id=nlm_gene_noph
# # no context
# python train.py train=nlm_gene.train.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene global_candidates=true exclude_context=true run_id=nlm_gene_noctx
# # biobert
python train.py train=nlm_gene.train.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene global_candidates=true run_id=nlm_gene_biobert lm=dmis-lab/biobert-v1.1 max_epochs=20 load_last_model=true

