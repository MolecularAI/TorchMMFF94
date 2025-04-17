from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from src.forcefield import TorchMMFF94


if __name__ == "__main__":
    device = "cpu"
    protein = Chem.MolFromPDBFile("data/5zcu/protein.pdb", sanitize=False, removeHs=False)
    protein = Chem.AddHs(protein, addCoords=True)
    Chem.GetSSSR(protein)

    ligand = Chem.SDMolSupplier("data/5zcu/ligand.sdf", sanitize=False, removeHs=False)[0]
    Chem.GetSSSR(ligand)
    
    ff = TorchMMFF94(protein=protein, device=device)
    ff.setup(ligand)
    x = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float32, device=device)
    x = x.requires_grad_(True)

    opt = torch.optim.Adam([x], lr=1e-3)
    for _ in range(20):
        opt.zero_grad()
        protein_ligand_energy, _ = ff.forward(x)
        print(f"Energy: {protein_ligand_energy.item():.3f}")
        protein_ligand_energy.backward()
        opt.step()
