import os

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from tqdm import tqdm
from typing import Dict
from utils.feat import (
    CircularFingerprint,
    DescriptorCalculator,
    CombinedFeaturizer,
    ChemBERTaFeaturizer,
)

DATA_ROOT = "s3://kin-del-2024/data"


def featurize(df, representation, smiles_col, label_col=None):
    """
    Featurize a DataFrame of molecules using the selected featurizer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing molecule data.
    representation : str
        The type of featurizer to use. Options: 'circular', 'descriptor', 'combined', 'chemberta'.
    smiles_col : str
        Name of the column containing SMILES strings.
    label_col : str, optional
        Name of the column containing labels (if available).

    Returns
    -------
    np.array
        Array of molecular features.
    np.array, optional
        Array of labels if label_col is provided.
    """
    if representation == "circular":
        featurizer = CircularFingerprint()
    elif representation == "descriptor":
        featurizer = DescriptorCalculator()
    elif representation == "combined":
        featurizer = CombinedFeaturizer(CircularFingerprint(), DescriptorCalculator())
    elif representation == "chemberta":
        featurizer = ChemBERTaFeaturizer()
    else:
        raise ValueError(f"Unknown representation: {representation}")

    fps = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(featurizer._featurize(mol))
        else:
            print(f"Invalid SMILES: {smiles}")
            fps.append(np.zeros((1,)))  # Placeholder for invalid molecules

    if label_col is not None:
        return np.array(fps), np.array(df[label_col])
    else:
        return np.array(fps)


def get_training_data(target, split_index, split_type):
    df = pd.read_parquet(os.path.join(DATA_ROOT, f"{target}_1M.parquet")).rename(
        {"target_enrichment": "y"}, axis="columns"
    )
    df_split = pd.read_parquet(
        os.path.join(DATA_ROOT, "splits", f"{target}_{split_type}.parquet")
    )
    return (
        df[df_split[f"split{split_index}"] == "train"],
        df[df_split[f"split{split_index}"] == "valid"],
        df[df_split[f"split{split_index}"] == "test"],
    )


def get_testing_data(target, in_library=False):
    data = {
        "on": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_ondna.csv"), index_col=0
        ).rename({"kd": "y"}, axis="columns"),
        "off": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_offdna.csv"), index_col=0
        ).rename({"kd": "y"}, axis="columns"),
    }
    if in_library:
        data["on"] = data["on"].dropna(subset="molecule_hash")
        data["off"] = data["off"].dropna(subset="molecule_hash")
    return data


def rmse(preds, target):
    return np.sqrt(np.mean((preds - target) ** 2))


def calculate_performance(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Calculate Spearman's rho and RMSE for predictions."""
    preds, uncertainties = model.predict(X_test, return_uncertainty=True)
    rho, _ = spearmanr(preds, y_test)
    errors = np.abs(preds - y_test)
    uncertainty_corr, _ = spearmanr(errors, uncertainties)
    return {
        "rho": rho,
        "rmse": rmse(preds, y_test),
        "uncertainty_corr": uncertainty_corr,
    }


def print_results(results: Dict):
    """Print results in a human-readable format."""
    print("\n### Results Summary ###\n")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key.capitalize()}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    print(f"  {subkey.capitalize()}:")
                    for metric, metric_value in subvalue.items():
                        print(f"    {metric}: {metric_value:.4f}")
                else:
                    print(f"  {subkey}: {subvalue:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    print("\n#######################\n")
