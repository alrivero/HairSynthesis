# CSE527 Introduction to Computer Vision (Fall 2025) - Final Project

**Project Title: Learning 2D Hair Orientation Maps through Analysis-by-Neural-Synthesis**

This codebased depends on [SMIRK](https://github.com/georgeretsi/smirk) and [HairStep](github.com/GAP-LAB-CUHK-SZ/HairStep).
We used [FFHQ256](https://github.com/NVlabs/ffhq-dataset) dataset for training and HiSa dataset from [HairStep](https://github.com/GAP-LAB-CUHK-SZ/HairStep) for evaluation.

For environment setup, you need to install the environment following SMIRK and HairStep environments.
Add HairStep folder to `smirk/external/HairStep`, it mostly similar to [HairStep](github.com/GAP-LAB-CUHK-SZ/HairStep), with some small changes to run on FFHQ256.

Steps:
1. Run hair mask extraction using `external/HairStep/scripts/img2masks.py`
2. Training commands are in `submit_jobs/cmds.md` or you can also use slurm scripts under `submit_jobs` to submit jobs

