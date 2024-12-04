#!/usr/bin/bash

/opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt --use_ms
/opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -b --use_ms
/opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -f --use_ms


# /opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -v 100 -s 6096 --use_ms
# /opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -b -v 100 -s 6096 --use_ms
/opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -f -v 100 -s 6096 --use_ms


# /opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -v 100 -s 3000 --use_ms
# /opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -b -v 100 -s 3000 --use_ms
# /opt/conda/envs/ptca/bin/python /scratch/nnscaler/experiments/minfer_phi/minfer_ops/fa_attn_test/fa_attn_profile.py -m mfmb_dt -f -v 100 -s 3000 --use_ms


