import sys
import subprocess


def main():
    ROOT_DIR = "/gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk/logs/smirkhair"
    result_dir_map = {
        'normmag_depthfrozen': "20251203_011904_normmag_depthfrozen",
        'normmag_nodepth': "20251203_012259_normmag_nodepth",
        'normmag_local_depthfrozen': "20251203_113333_normmag_local_depthfrozen",
        'normmag_local_nodepth': "20251203_124425_normmag_local_nodepth"
    }
    subsets = ['val', 'test']
    stress_config = [
        ['blur'],
        # ['noise', 'noise_sigma', 0.1],
        # ['noise', 'noise_sigma', 0.5],
    ]

    for k, v in result_dir_map.items():
        for subset in subsets:
            for cfg in stress_config:
                cmd = [
                    sys.executable,     # use the python from the currently activated conda env
                    "gen_stress.py",
                    f"configs/config_train_hs_{k}.yaml",
                    f"resume=logs/smirkhair/{v}/model_49.pt",
                    "--subset", subset,
                    "--stress", cfg[0],
                ]

                if len(cfg) > 1:
                    param_name = cfg[1]
                    param_value = str(cfg[2])
                    cmd.extend([f"--{param_name}", param_value])
                
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()