from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pytest
import torch

from src.forcefield import TorchMMFF94


IBUPROFEN = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
PENICILLIN = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"


@pytest.mark.parametrize("smiles", [IBUPROFEN, PENICILLIN])
@pytest.mark.parametrize("seed", [1234, 5678])
def test_bucketsize_dataset(smiles, seed):
    # dataset_path = Path(__file__).parent / "dataset/mols"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    ff1 = TorchMMFF94(device="cpu")
    ff1.setup(mol)
    x = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    protein_ligand_energy, ligand_energy = ff1.forward(x)

    prop = AllChem.MMFFGetMoleculeProperties(mol)
    ff2 = AllChem.MMFFGetMoleculeForceField(mol, prop)

    assert np.allclose(ff2.CalcEnergy(), ligand_energy, atol=1e-3)