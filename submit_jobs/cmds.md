

cd /gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk
sbatch submit_jobs/train_job_long.slurm
sbatch submit_jobs/train_job.slurm


## evaluation
```bash
python visualize.py "logs/smirkhair/20251127_184747_org/model_48.pt"
python visualize.py "logs/smirkhair/20251129_002446_normmag_local_depthfrozen/model_49.pt"
python visualize.py "logs/smirkhair/20251129_002459_normmag_local_nodepth/model_49.pt"
python visualize.py "logs/smirkhair/20251129_224026_normmag_depthfrozen/model_49.pt"
python visualize.py "logs/smirkhair/20251129_123124_normmag_nodepth/model_49.pt"
```

## `gen_stress.py`
```bash
python gen_stress.py configs/config_train_hs_normmag_depthfrozen.yaml resume='logs/smirkhair/20251203_113333_normmag_local_depthfrozen/model_49.pt'
```

