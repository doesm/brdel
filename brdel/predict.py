import os
import yaml
import argparse
from typing import List, Optional, Dict
from utils.data import (
    get_training_data,
    get_testing_data,
    calculate_performance,
    print_results,
)

from models.br import BayesianRidgeModel
import numpy as np
import random


def training_subjob(
    output_dir: str,
    split_index: int,
    split_type: str,
    target: str,
    representation: str,
    hyperparameters: Optional[Dict] = None,
) -> Dict:
    """
    Runs a training subjob for a specific split, target, and representation.

    Args:
        output_dir (str): Directory to save the results.
        split_index (int): Index of the data split.
        split_type (str): Type of split ('random', 'disynthon').
        target (str): The target for model training.
        representation (str): Molecular representation to use (e.g., 'circular', 'chemberta').
        hyperparameters (Optional[Dict]): Dictionary of hyperparameters for the model.

    Returns:
        Dict: Dictionary containing the performance results on test datasets.
    """
    random.seed(123)
    np.random.seed(123)

    # Get the training, validation, and test data
    df_train, df_valid, df_test = get_training_data(target, split_index, split_type)

    # Use default hyperparameters if none provided
    hyperparameters = hyperparameters or {}

    # Instantiate the model
    model = BayesianRidgeModel(**hyperparameters)

    # Prepare the dataset using the specified representation
    data = model.prepare_dataset(df_train, df_valid, df_test, representation)

    # Train the model
    model.train()

    # Evaluate performance
    results = {"test": calculate_performance(model, data.test.x, data.test.y)}

    # Evaluate on additional datasets
    for dataset_name, in_library in [("extended", False), ("in_library", True)]:
        testing_data = get_testing_data(target, in_library=in_library)
        results[dataset_name] = {}
        for condition, dataset in testing_data.items():
            X_test, y_test = model.featurize(dataset, representation)
            results[dataset_name][condition] = calculate_performance(
                model, X_test, y_test
            )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(
        output_dir, f"results_{split_type}_s{split_index}_{target}.yml"
    )
    with open(results_path, "w") as fp:
        yaml.dump(results, fp)

    # Print results
    print(f"Results saved to {results_path}")
    print_results(results)


def train(
    output_dir: str,
    targets: List[str],
    splits: List[str],
    split_indexes: List[int],
    representation: str,
    hyperparameters: Optional[str] = None,
):
    """
    Trains models on multiple targets, splits, and split indexes.

    Args:
        output_dir (str): Directory to save the results.
        targets (List[str]): List of targets to train on.
        splits (List[str]): List of data split types ('random', 'disynthon').
        split_indexes (List[int]): List of split indexes to process.
        representation (str): Molecular representation (e.g., 'circular', 'chemberta').
        hyperparameters (Optional[str]): Path to a YAML file containing hyperparameters.

    Returns:
        List[str]: List of file paths to the saved results.
    """
    # Load hyperparameters if provided
    hyperparams = None
    if hyperparameters:
        with open(hyperparameters, "r") as hp_file:
            hyperparams = yaml.safe_load(hp_file)

    results_files = []
    for split_index in split_indexes:
        for target in targets:
            for split_type in splits:
                result = training_subjob(
                    output_dir=output_dir,
                    split_index=split_index,
                    split_type=split_type,
                    target=target,
                    representation=representation,
                    hyperparameters=hyperparams,
                )
                results_files.append(result)
    return results_files


if __name__ == "__main__":
    """
    Main entry point for the script. Parses command-line arguments and starts training.
    """
    parser = argparse.ArgumentParser(
        description="Train models on specified data splits and targets."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save the training results."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["ddr1", "mapk14"],
        help="List of targets to train on.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "disynthon"],
        help="List of data splits.",
    )
    parser.add_argument(
        "--split-indexes",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="List of split indexes.",
    )
    parser.add_argument(
        "--representation",
        default="morgan",
        help="Molecular representation to use ('circular', 'descriptor', 'combined').",
    )
    parser.add_argument(
        "--hyperparameters",
        help="Path to YAML file containing hyperparameters (optional).",
    )

    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        targets=args.targets,
        splits=args.splits,
        split_indexes=args.split_indexes,
        representation=args.representation,
        hyperparameters=args.hyperparameters,
    )
