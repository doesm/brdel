import argparse
import os
import numpy as np
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    for target in ("mapk14", "ddr1"):
        print(target)
        for split_type in ("random", "disynthon"):
            print(split_type)
            results_dir = args.model_path
            # if not os.path.exists(results_dir):
            #     print(f"Split {split_type} not found. Skipping.")
            #     continue

            results = {
                "test": {"rho": [], "rmse": [], "uncertainty_corr": []},
                "extended": {
                    "on": {"rho": [], "uncertainty_corr": []},
                    "off": {"rho": [], "uncertainty_corr": []},
                },
                "in_library": {
                    "on": {"rho": [], "uncertainty_corr": []},
                    "off": {"rho": [], "uncertainty_corr": []},
                },
            }

            for split_index in range(1, 6):
                out_file = os.path.join(
                    results_dir, f"results_{split_type}_s{split_index}_{target}.yml"
                )
                if not os.path.exists(out_file):
                    continue

                with open(out_file, "r") as file:
                    content = yaml.load(file, Loader=yaml.Loader)
                    for test_set in ("test", "extended", "in_library"):
                        if test_set == "test":
                            results[test_set]["rho"].append(content[test_set]["rho"])
                            results[test_set]["rmse"].append(content[test_set]["rmse"])
                            results[test_set]["uncertainty_corr"].append(
                                content[test_set]["uncertainty_corr"]
                            )
                        else:
                            for condition in ("on", "off"):
                                results[test_set][condition]["rho"].append(
                                    content[test_set][condition]["rho"]
                                )
                                results[test_set][condition]["uncertainty_corr"].append(
                                    content[test_set][condition]["uncertainty_corr"]
                                )

            for test_set in ("test", "in_library", "extended"):
                if test_set == "test":
                    metric = "rmse"
                    scores = [score ** 2 for score in results[test_set][metric]]
                    mu = np.mean(scores)
                    std = np.std(scores)
                    print(
                        f"{test_set.capitalize()} {metric}: Mean = {mu:.3f}, Std = {std:.3f}"
                    )
                else:
                    for condition in ("on", "off"):
                        metric = "rho"
                        mu = -np.mean(results[test_set][condition][metric])
                        std = np.std(results[test_set][condition][metric])
                        uncertainty_corr = np.mean(
                            results[test_set][condition]["uncertainty_corr"]
                        )
                        print(
                            f"{test_set.capitalize()} {condition} {metric}: Mean = {mu:.3f}, "
                            f"Std = {std:.3f}, Uncertainty_corr = {uncertainty_corr:.3f}"
                        )
