

cd /gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk
sbatch submit_jobs/train_job_long.slurm

sbatch submit_jobs/train_job.slurm


## running
sbatch submit_jobs/train_job_org.slurm                              # running
sbatch submit_jobs/train_job_normmag_nodepth.slurm                  # done
sbatch submit_jobs/train_job_normmag_depthfrozen.slurm              # done

sbatch submit_jobs/train_job_normmag_local_nodepth.slurm        
sbatch submit_jobs/train_job_normmag_local_depthfrozen.slurm        # done


## notes
logs/smirkhair/20251203_010438          org                         # error
logs/smirkhair/20251203_011904          normmag_depthfrozen         # done
logs/smirkhair/20251203_012259          normmag_nodepth             # done

logs/smirkhair/20251203_021034          org - error
logs/smirkhair/20251203_022207          org - error
logs/smirkhair/20251203_022441          org                         # done
logs/smirkhair/20251203_113333          normmag_local_depthfrozen   # done
logs/smirkhair/20251203_124425          normmag_local_nodepth       # done

# evaluation
```bash
python visualize.py "logs/smirkhair/20251127_184747_org/model_48.pt"

# python visualize.py "logs/smirkhair/20251129_002446_normmag_local_depthfrozen_bk/model_49.pt"
python visualize.py "logs/smirkhair/20251129_002446_normmag_local_depthfrozen/model_49.pt"

python visualize.py "logs/smirkhair/20251129_002459_normmag_local_nodepth/model_49.pt"
python visualize.py "logs/smirkhair/20251129_224026_normmag_depthfrozen/model_49.pt"
python visualize.py "logs/smirkhair/20251129_123124_normmag_nodepth/model_49.pt"
```