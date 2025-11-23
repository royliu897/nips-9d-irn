Utilizes a Siren layer based irn to predict scattering energy intensity values, with
a FiLM modulation to better generalize across Hamiltonians.

Original Siren paper: https://arxiv.org/abs/2006.09661
Original FiLM paper: https://arxiv.org/abs/1709.07871

Workflow:
1) nips_9d - Generates data
2) curate.py - Converts data to .pt
3) run_train.slurm - runs train_9d.py using the config in train_config_9d.py
4) dense_slice.jl - generates dense data for ground truth image
5) vis_9d.py - generates plot and png comparing ground truth to predicted

WandB: https://wandb.ai/royliu2007-university-of-texas-at-austin/nips-9d-siren-film?nw=nwuserroyliu2007
