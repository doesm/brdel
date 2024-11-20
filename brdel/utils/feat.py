import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from tqdm import tqdm
from rdkit.Chem import Descriptors

from transformers import AutoTokenizer, AutoModel
import torch


"""Module copied over from DeepChem which is under a MIT license
https://github.com/deepchem/deepchem/blob/master/deepchem/feat/base_classes.py"""


class Featurizer(object):
    """
    Abstract class for calculating a set of features for a molecule.
    Child classes implement the _featurize method for calculating features
    for a single molecule.
    """

    def featurize(
        self, mols: list[Chem.Mol], use_tqdm: bool = True, asarray: bool = True
    ):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        use_tqdm: bool
            Whether a progress bar will be printed in the featurization.
        asarray: bool
            return featurized data as a numpy array (if False, return a list).
        """
        mols = [mols] if isinstance(mols, Chem.Mol) else mols
        features = []
        if use_tqdm:
            mols_iterable = tqdm(mols)
        else:
            mols_iterable = mols
        for mol in mols_iterable:
            if mol is not None:
                features.append(self._featurize(mol))
            else:
                features.append(np.array([]))

        if asarray:
            return np.asarray(features)
        return features

    def _featurize(self, mol: Chem.Mol):
        """
        Calculate features for a single molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        raise NotImplementedError("Featurizer is not defined.")

    def __call__(
        self, mols: list[Chem.Mol], use_tqdm: bool = True, asarray: bool = True
    ):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        use_tqdm: bool
            Whether a progress bar will be printed in the featurization.
        asarray: bool
            return featurized data as a numpy array (if False, return a list).
        """
        return self.featurize(mols, use_tqdm=use_tqdm, asarray=asarray)


class CircularFingerprint(Featurizer):
    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_bond_type: bool = True,
        as_numpy_array: bool = True,
        **kwargs,
    ):
        """
        Interface for MorganFingerprints

        Parameters
        ----------
        radius: int, (default 2)
            Radius of graph to consider
        n_bits: int, (default 2048)
            Number of bits in the fingerprint
        use_chirality: bool, (default False)
            Whether to consider chirality in fingerprint generation
        use_bond_type: bool, (default True)
            Whether to consider bond ordering in the fingerprint generation
        as_numpy_array: bool, (default True)
            Whether or not to return as numpy array

        """
        self.as_numpy_array = as_numpy_array
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=use_chirality,
            useBondTypes=use_bond_type,
        )
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        fingerprint = self.mfpgen.GetFingerprint(mol)
        if self.as_numpy_array:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, arr)
        else:
            arr = fingerprint
        return arr


class DescriptorCalculator(Featurizer):
    def __init__(self, descriptors=None):
        """
        Initializes descriptor calculator.

        Parameters
        ----------
        descriptors: list of callable
            List of descriptor functions from RDKit (e.g., Descriptors.MolWt).
        """
        if descriptors is None:
            self.descriptors = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.qed]
        else:
            self.descriptors = descriptors
        super().__init__()

    def _featurize(self, mol: Chem.Mol):
        return np.array([desc(mol) for desc in self.descriptors])


class CombinedFeaturizer(Featurizer):
    def __init__(self, fingerprint_featurizer, descriptor_featurizer):
        """
        Combines fingerprint and descriptor featurizers.

        Parameters
        ----------
        fingerprint_featurizer: Featurizer
            Instance of a fingerprint featurizer (e.g., CircularFingerprint).
        descriptor_featurizer: Featurizer
            Instance of a descriptor featurizer (e.g., DescriptorCalculator).
        """
        self.fingerprint_featurizer = fingerprint_featurizer
        self.descriptor_featurizer = descriptor_featurizer
        super().__init__()

    def _featurize(self, mol: Chem.Mol):
        fingerprint = self.fingerprint_featurizer._featurize(mol)
        descriptors = self.descriptor_featurizer._featurize(mol)
        return np.concatenate((fingerprint, descriptors))


class ChemBERTaFeaturizer(Featurizer):
    def __init__(
        self, model_name: str = "DeepChem/ChemBERTa-10M-MTR"
    ):  # seyonec/ChemBERTa-zinc-base-v1
        """
        Featurizer using the ChemBERTa model.

        Parameters
        ----------
        model_name : str, default 'DeepChem/ChemBERTa-10M-MTR'
            Name of the pre-trained ChemBERTa model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        super().__init__()

    def _featurize(self, mol: Chem.Mol):
        smiles = Chem.MolToSmiles(mol)
        inputs = self.tokenizer(
            smiles, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding as a representation
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
