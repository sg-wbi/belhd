#!/usr/bin/env bash
#
EXP_DIR="/vol/fob-wbia-vol1/wbi/gardasam/data/belb-dr/belhd_ablations"

# # no global candidates
# python predict.py test=nlm_gene.test.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene run_id=nlm_gene_nocs
# # no projection layer
# python predict.py test=nlm_gene.test.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene global_candidates=true project=-1 run_id=nlm_gene_noph
# # no context
# python predict.py test=nlm_gene.test.gene exp_dir="$EXP_DIR" kb_subset=nlm_gene global_candidates=true exclude_context=true run_id=nlm_gene_noctx
