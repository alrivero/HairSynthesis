#!/bin/bash

# # stress test 1: gaussian blur

# python gen_stress.py configs/config_train_hs_normmag_depthfrozen.yaml resume='logs/smirkhair/20251203_011904_normmag_depthfrozen/model_49.pt' --subset test
# python gen_stress.py configs/config_train_hs_normmag_depthfrozen.yaml resume='logs/smirkhair/20251203_011904_normmag_depthfrozen/model_49.pt' --subset val

# python gen_stress.py configs/config_train_hs_normmag_nodepth.yaml resume='logs/smirkhair/20251203_012259_normmag_nodepth/model_49.pt' --subset test
# python gen_stress.py configs/config_train_hs_normmag_nodepth.yaml resume='logs/smirkhair/20251203_012259_normmag_nodepth/model_49.pt' --subset val

# python gen_stress.py configs/config_train_hs_normmag_local_depthfrozen.yaml resume='logs/smirkhair/20251203_113333_normmag_local_depthfrozen/model_49.pt' --subset test
# python gen_stress.py configs/config_train_hs_normmag_local_depthfrozen.yaml resume='logs/smirkhair/20251203_113333_normmag_local_depthfrozen/model_49.pt' --subset val

# python gen_stress.py configs/config_train_hs_normmag_local_nodepth.yaml resume='logs/smirkhair/20251203_124425_normmag_local_nodepth/model_49.pt' --subset test
# python gen_stress.py configs/config_train_hs_normmag_local_nodepth.yaml resume='logs/smirkhair/20251203_124425_normmag_local_nodepth/model_49.pt' --subset val


# stress test 2: gaussian noise

# python gen_stress.py configs/config_train_hs_normmag_depthfrozen.yaml resume='logs/smirkhair/20251203_011904_normmag_depthfrozen/model_49.pt' --subset test --stress noise --noise_sigma 0.5

python gen_stress.py configs/config_train_hs_normmag_depthfrozen.yaml resume='logs/smirkhair/20251203_011904_normmag_depthfrozen/model_49.pt' --subset val --stress noise --noise_sigma 0.5

python gen_stress.py configs/config_train_hs_normmag_nodepth.yaml resume='logs/smirkhair/20251203_012259_normmag_nodepth/model_49.pt' --subset test --stress noise --noise_sigma 0.5
# python gen_stress.py configs/config_train_hs_normmag_nodepth.yaml resume='logs/smirkhair/20251203_012259_normmag_nodepth/model_49.pt' --subset val --stress noise --noise_sigma 0.5

# python gen_stress.py configs/config_train_hs_normmag_local_depthfrozen.yaml resume='logs/smirkhair/20251203_113333_normmag_local_depthfrozen/model_49.pt' --subset test --stress noise --noise_sigma 0.5
python gen_stress.py configs/config_train_hs_normmag_local_depthfrozen.yaml resume='logs/smirkhair/20251203_113333_normmag_local_depthfrozen/model_49.pt' --subset val --stress noise --noise_sigma 0.5

# python gen_stress.py configs/config_train_hs_normmag_local_nodepth.yaml resume='logs/smirkhair/20251203_124425_normmag_local_nodepth/model_49.pt' --subset test --stress noise --noise_sigma 0.5
python gen_stress.py configs/config_train_hs_normmag_local_nodepth.yaml resume='logs/smirkhair/20251203_124425_normmag_local_nodepth/model_49.pt' --subset val --stress noise --noise_sigma 0.5
