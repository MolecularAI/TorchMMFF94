import itertools

from rdkit import Chem
from rdkit.Chem import AllChem
import torch


MDYNE_A_TO_KCAL_MOL = 143.9325


# https://www.charmm-gui.org/charmmdoc/mmff.html
class TorchMMFF94:
    def __init__(self, protein=None, th=1.5, device="cpu"):
        self.device = device
        self.th = th
        self.protein = protein
        if self.protein:
            self.X = torch.tensor(
                protein.GetConformer().GetPositions(),
                dtype=torch.float32,
                device=self.device,
            )
        try:
            self.protein_prop = AllChem.MMFFGetMoleculeProperties(protein)
            # A bit ugly but efficient
            m = self.protein.GetNumAtoms()
            self.protein_eps_vdw = torch.zeros(
                (m,), dtype=torch.float32, device=self.device
            )
            self.protein_r_dist_vdw = 1e-3 + torch.zeros(
                (m,), dtype=torch.float32, device=self.device
            )
            self.protein_Q_ele = torch.zeros(
                (m,), dtype=torch.float32, device=self.device
            )

            for j in range(m):
                _, _, _rdistj, _epsj = self.protein_prop.GetMMFFVdWParams(j, j)
                self.protein_eps_vdw[j] = _epsj
                self.protein_r_dist_vdw[j] = _rdistj
                self.protein_Q_ele[j] = self.protein_prop.GetMMFFPartialCharge(j)
        except:
            self.protein_prop = None
        self.params = {}

    def setup(self, mol):
        self.mol = mol
        self.ff_prop = AllChem.MMFFGetMoleculeProperties(mol)
        keys = list(self.params.keys())
        for k in keys:
            del self.params[k]
        self._e_vdw_setup()
        self._e_ele_setup()
        self._e_bond_setup()
        self._e_angle_setup()
        self._e_oop_setup()
        self._e_stretch_bend_setup()
        self._e_torsion_setup()
        if self.protein_prop is not None:
            self._e_vdw_ligand_protein_setup()
            self._e_ele_ligand_protein_setup()

    def _e_vdw_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        n = mol.GetNumAtoms()
        eps = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        r_dist = 1e-3 + torch.zeros((n, n), dtype=torch.float32, device=self.device)
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                path = Chem.GetShortestPath(mol, i, j)
                _rdist_un, _eps_un, _rdist, _eps = ff_prop.GetMMFFVdWParams(i, j)

                if len(path) < 4:
                    _eps = 0.0
                # if dist[i, j] > 100:
                #     _eps = 0.0
                # elif len(path) > 4:
                #     _rdist = _rdist_un
                #     _eps = _eps_un
                eps[i, j] = _eps
                r_dist[i, j] = _rdist
                eps[j, i] = _eps
                r_dist[j, i] = _rdist
        self.params["e_vdw"] = [eps, r_dist]

    def _e_ele_setup(self):
        sc1_4 = 0.75
        mol, ff_prop = self.mol, self.ff_prop
        n = mol.GetNumAtoms()
        Q = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        den = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        for i in range(mol.GetNumAtoms()):
            qi = ff_prop.GetMMFFPartialCharge(i)
            for j in range(i + 1, mol.GetNumAtoms()):
                qj = ff_prop.GetMMFFPartialCharge(j)
                path = Chem.GetShortestPath(mol, i, j)
                Q[i, j] = qi * qj
                if len(path) == 4:
                    den[i, j] = sc1_4
                elif len(path) > 4:
                    den[i, j] = 1.0
        self.params["e_ele"] = [Q, den]

    def _e_bond_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        n = mol.GetNumAtoms()
        kbond = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        r_dist = 1e-3 + torch.zeros((n, n), dtype=torch.float32, device=self.device)
        mask = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            btype, _kbond, _rdist = ff_prop.GetMMFFBondStretchParams(mol, i, j)

            # if len(path) < 4:
            #     _eps = 0.0
            # if dist[i, j] > 100:
            #     _eps = 0.0
            # elif len(path) > 4:
            #     _rdist = _rdist_un
            #     _eps = _eps_un
            kbond[i, j] = _kbond
            r_dist[i, j] = _rdist
            kbond[j, i] = _kbond
            r_dist[j, i] = _rdist
            mask[i, j] = 1
        self.params["e_bond"] = [kbond, r_dist, mask]

    def _e_angle_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        idx = []
        kbonds = []
        theta0s = []
        for p in Chem.FindAllPathsOfLengthN(
            mol, 3, useBonds=False, useHs=True, onlyShortestPaths=True
        ):
            idx.append(tuple(p))
            angletype, kbond, theta0 = ff_prop.GetMMFFAngleBendParams(mol, *p)
            kbonds.append(kbond)
            theta0s.append(theta0)

        idx = torch.tensor(idx, device=self.device)
        kbonds = torch.tensor(kbonds, dtype=torch.float32, device=self.device)
        theta0s = torch.tensor(theta0s, dtype=torch.float32, device=self.device)
        self.params["e_angle"] = [idx, kbonds, theta0s]

    def _e_oop_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        valid_quadruples = []
        koops = []
        for atom in mol.GetAtoms():
            center_idx = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

            if len(neighbors) >= 3:
                for triplet in itertools.combinations(neighbors, 3):
                    for perm in itertools.permutations(triplet):
                        quadruple = (perm[0], center_idx, perm[1], perm[2])
                        koop = ff_prop.GetMMFFOopBendParams(mol, *quadruple)
                        if koop is not None:
                            if quadruple not in valid_quadruples:
                                valid_quadruples.append(quadruple)
                                koops.append(koop)

        valid_quadruples = torch.tensor(valid_quadruples, device=self.device)
        koops = torch.tensor(koops, dtype=torch.float32, device=self.device)
        self.params["e_oop"] = [valid_quadruples, koops]

    def _e_stretch_bend_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        idx = []
        kbaIJK, kbaKJI = [], []
        rij, rkj = [], []
        theta_ijk = []

        for p in Chem.FindAllPathsOfLengthN(
            mol, 3, useBonds=False, useHs=True, onlyShortestPaths=False
        ):
            i, j, k = tuple(p)
            params1 = ff_prop.GetMMFFStretchBendParams(mol, i, j, k)
            params2 = ff_prop.GetMMFFAngleBendParams(mol, i, j, k)
            params3 = ff_prop.GetMMFFBondStretchParams(mol, i, j)
            params4 = ff_prop.GetMMFFBondStretchParams(mol, j, k)

            if (
                (params1 is not None)
                and (params2 is not None)
                and (params3 is not None)
                and (params4 is not None)
            ):
                _stretch_bend_type, _kbaIJK, _kbaKJI = params1
                _, _, _theta0 = params2
                _, _, _rij = params3
                _, _, _rkj = params4
                idx.append((i, j, k))
                kbaIJK.append(_kbaIJK)
                kbaKJI.append(_kbaKJI)
                rij.append(_rij)
                rkj.append(_rkj)
                theta_ijk.append(torch.pi / 180 * _theta0)

        idx = torch.tensor(idx, device=self.device)
        kbaIJK = torch.tensor(kbaIJK, dtype=torch.float32, device=self.device)
        kbaKJI = torch.tensor(kbaKJI, dtype=torch.float32, device=self.device)
        rij = torch.tensor(rij, dtype=torch.float32, device=self.device)
        rkj = torch.tensor(rkj, dtype=torch.float32, device=self.device)
        theta_ijk = torch.tensor(theta_ijk, dtype=torch.float32, device=self.device)
        self.params["e_stretch_bend"] = [idx, kbaIJK, kbaKJI, rij, rkj, theta_ijk]

    def _e_torsion_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        idx = []
        V1, V2, V3 = [], [], []
        # for p in Chem.FindAllPathsOfLengthN(mol, 4, useBonds=False, useHs=True, onlyShortestPaths=True):
        #     params = ff_prop.GetMMFFTorsionParams(mol, *p)
        #     # if (params is not None) and (len(set(p)) == 4) and (tuple(sorted(p)) not in frontier):
        #     if (params is not None):
        #         _, _V1, _V2, _V3 = params
        #         V1.append(_V1)
        #         V2.append(_V2)
        #         V3.append(_V3)
        #         idx.append(tuple(p))
        for bond in mol.GetBonds():
            j = bond.GetBeginAtom()
            k = bond.GetEndAtom()
            j_idx = j.GetIdx()
            k_idx = k.GetIdx()
            for i in j.GetNeighbors():
                i_idx = i.GetIdx()
                if i_idx == k_idx:
                    continue
                for l in k.GetNeighbors():
                    l_idx = l.GetIdx()
                    if l_idx == j_idx:
                        continue
                    p = (i_idx, j_idx, k_idx, l_idx)
                    params = ff_prop.GetMMFFTorsionParams(mol, *p)
                    if params is not None:
                        _, _V1, _V2, _V3 = params
                        V1.append(_V1)
                        V2.append(_V2)
                        V3.append(_V3)
                        idx.append(p)

        idx = torch.tensor(idx, device=self.device)
        V1 = torch.tensor(V1, dtype=torch.float32, device=self.device)
        V2 = torch.tensor(V2, dtype=torch.float32, device=self.device)
        V3 = torch.tensor(V3, dtype=torch.float32, device=self.device)
        self.params["e_torsion"] = [idx, V1, V2, V3]

    def _e_vdw_ligand_protein_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        protein = self.protein
        n = mol.GetNumAtoms()
        m = protein.GetNumAtoms()
        eps = torch.zeros((n, m), dtype=torch.float32, device=self.device)
        r_dist = 1e-3 + torch.zeros((n, m), dtype=torch.float32, device=self.device)

        for i in range(n):
            _, _, _rdisti, _epsi = ff_prop.GetMMFFVdWParams(i, i)
            eps[i] = (_epsi * self.protein_eps_vdw) ** 0.5
            r_dist[i] = _rdisti * 0.5 + self.protein_r_dist_vdw * 0.5

        self.params["e_vdw_ligand_protein"] = [eps, r_dist]

    def _e_ele_ligand_protein_setup(self):
        mol, ff_prop = self.mol, self.ff_prop
        protein = self.protein
        sc1_4 = 0.75
        n = mol.GetNumAtoms()
        m = protein.GetNumAtoms()
        Q = torch.zeros((n, m), dtype=torch.float32, device=self.device)
        for i in range(n):
            Q[i] = ff_prop.GetMMFFPartialCharge(i) * self.protein_Q_ele
        self.params["e_ele_ligand_protein"] = [Q]

    # Non-bound interaction
    def e_vdw(self, dist):
        eps, r_dist = self.params["e_vdw"]
        mask = (dist <= 100).float()
        eps = eps * mask
        vdw1 = 1.07
        vdw1m1 = vdw1 - 1.0
        vdw2 = 1.12
        vdw2m1 = vdw2 - 1.0
        dist2 = dist * dist
        dist7 = dist2 * dist2 * dist2 * dist
        aTerm = vdw1 * r_dist / (dist + vdw1m1 * r_dist + 1e-8)
        aTerm2 = aTerm * aTerm
        aTerm7 = aTerm2 * aTerm2 * aTerm2 * aTerm
        R_star_ij2 = r_dist * r_dist
        R_star_ij7 = R_star_ij2 * R_star_ij2 * R_star_ij2 * r_dist
        bTerm = vdw2 * R_star_ij7 / (dist7 + vdw2m1 * R_star_ij7 + 1e-8) - 2.0
        res = eps * aTerm7 * bTerm
        i, j = torch.triu_indices(res.size(0), res.size(1), offset=1) #, device=res.device)
        return res[i, j].sum()

    # Non-bound interaction
    def e_ele(self, dist):
        Q, den = self.params["e_ele"]
        corr_dist = dist + 0.05
        diel = 332.0716
        # if (dielModel == RDKit::MMFF::DISTANCE)
        #     corr_dist *= corr_dist
        res = diel * Q / corr_dist * den
        i, j = torch.triu_indices(res.size(0), res.size(1), offset=1) #, device=res.device)
        return res[i, j].sum()

    def e_bond(self, dist):
        kbond, r_dist, mask = self.params["e_bond"]

        delta_r = dist - r_dist
        delta_r2 = delta_r * delta_r

        c1 = MDYNE_A_TO_KCAL_MOL
        cs = -2.0
        c3 = 7.0 / 12.0
        res = (
            0.5 * c1 * kbond * delta_r2 * (1.0 + cs * delta_r + c3 * cs * cs * delta_r2)
        )
        return (res * mask).sum()

    def e_angle(self, x):
        idx, kbonds, theta0s = self.params["e_angle"]

        p1 = x[idx[..., 0]]
        p2 = x[idx[..., 1]]
        p3 = x[idx[..., 2]]
        p12 = p1 - p2
        p32 = p3 - p2
        dist1 = ((p12**2).sum(-1) + 1e-8) ** 0.5
        dist2 = ((p32**2).sum(-1) + 1e-8) ** 0.5
        cos_theta = (p12 * p32).sum(-1) / dist1 / dist2
        theta = torch.arccos(torch.clip(cos_theta, -1.0, 1.0))
        angle = 180.0 / torch.pi * theta - theta0s
        cb = -0.006981317
        c2 = MDYNE_A_TO_KCAL_MOL * torch.pi / 180.0 * torch.pi / 180.0
        # if is_linear:
        #     res = MDYNE_A_TO_KCAL_MOL * ka * (1.0 + cosTheta);
        res = 0.5 * c2 * kbonds * angle * angle * (1.0 + cb * angle)
        return res.sum()

    def e_oop(self, x):
        valid_quadruples, koops = self.params["e_oop"]
        # ijkl
        rji = x[valid_quadruples[..., 0]] - x[valid_quadruples[..., 1]]
        rjk = x[valid_quadruples[..., 2]] - x[valid_quadruples[..., 1]]
        rjl = x[valid_quadruples[..., 3]] - x[valid_quadruples[..., 1]]

        rji = rji / torch.sqrt(torch.sum(rji**2, -1, keepdims=True) + 1e-8)
        rjk = rjk / torch.sqrt(torch.sum(rjk**2, -1, keepdims=True) + 1e-8)
        rjl = rjl / torch.sqrt(torch.sum(rjl**2, -1, keepdims=True) + 1e-8)

        n = torch.cross(rji, rjk, dim=-1)
        n = n / torch.sqrt(torch.sum(n**2, -1, keepdims=True) + 1e-8)
        sin_chi = (n * rjl).sum(-1)
        sin_chi = torch.clip(sin_chi, -1.0, 1.0)

        c2 = MDYNE_A_TO_KCAL_MOL * torch.pi / 180.0 * torch.pi / 180.0
        chi = 180.0 / torch.pi * torch.arcsin(sin_chi)
        # for some reason we count twice...
        # There is something in combination + permutations
        return (0.5 * c2 * koops * chi * chi).sum() / 2.0

    def e_stretch_bend(self, x):
        idx, kbaIJK, kbaKJI, rij, rkj, theta_ijk = self.params["e_stretch_bend"]

        dij = x[idx[..., 0]] - x[idx[..., 1]]
        dkj = x[idx[..., 2]] - x[idx[..., 1]]

        den1 = ((dij**2).sum(-1) + 1e-8) ** 0.5
        den2 = ((dkj**2).sum(-1) + 1e-8) ** 0.5

        cos_thetas = (dij * dkj).sum(-1) / den1 / den2
        cos_thetas = torch.clip(cos_thetas, -1.0, 1.0)
        angles = torch.arccos(cos_thetas)
        factor = MDYNE_A_TO_KCAL_MOL * (angles - theta_ijk)

        e = factor * (kbaIJK * (den1 - rij) + kbaKJI * (den2 - rkj))
        return e.sum()

    def e_torsion(self, x):
        idx, V1, V2, V3 = self.params["e_torsion"]
        r1 = x[idx[..., 0]] - x[idx[..., 1]]
        r2 = x[idx[..., 2]] - x[idx[..., 1]]
        r3 = x[idx[..., 1]] - x[idx[..., 2]]
        r4 = x[idx[..., 3]] - x[idx[..., 2]]

        t1 = torch.cross(r1, r2, dim=-1)
        t2 = torch.cross(r3, r4, dim=-1)

        t1_norm = (torch.sum(t1**2, -1) + 1e-8) ** 0.5
        t2_norm = (torch.sum(t2**2, -1) + 1e-8) ** 0.5

        cos_phi = (t1 * t2).sum(-1) / (t1_norm * t2_norm)
        cos_phi = torch.clip(cos_phi, -1.0, 1.0)

        cos2_phi = 2.0 * cos_phi * cos_phi - 1.0
        cos3_phi = cos_phi * (2.0 * cos2_phi - 1.0)
        res = 0.5 * (
            V1 * (1.0 + cos_phi) + V2 * (1.0 - cos2_phi) + V3 * (1.0 + cos3_phi)
        )
        return res.sum()

    def e_vdw_ligand_protein(self, dist):
        eps, r_dist = self.params["e_vdw_ligand_protein"]
        mask = (dist <= 100).float()
        eps = eps * mask
        vdw1 = 1.07
        vdw1m1 = vdw1 - 1.0
        vdw2 = 1.12
        vdw2m1 = vdw2 - 1.0
        dist2 = dist * dist
        dist7 = dist2 * dist2 * dist2 * dist
        aTerm = vdw1 * r_dist / (dist + vdw1m1 * r_dist + 1e-8)
        aTerm2 = aTerm * aTerm
        aTerm7 = aTerm2 * aTerm2 * aTerm2 * aTerm
        R_star_ij2 = r_dist * r_dist
        R_star_ij7 = R_star_ij2 * R_star_ij2 * R_star_ij2 * r_dist
        bTerm = vdw2 * R_star_ij7 / (dist7 + vdw2m1 * R_star_ij7 + 1e-8) - 2.0
        res = eps * aTerm7 * bTerm
        return res.sum()

    def e_ele_ligand_protein(self, dist):
        Q = self.params["e_ele_ligand_protein"][0]
        mask = (dist <= 100).float()
        corr_dist = dist + 0.05
        diel = 332.0716
        # if (dielModel == RDKit::MMFF::DISTANCE)
        #     corr_dist *= corr_dist
        res = diel * Q / corr_dist * mask
        return res.sum()

    def forward(self, x):
        protein = self.protein
        protein_prop = self.protein_prop

        dist = (torch.sum((x[:, None] - x[None]) ** 2, -1) + 1e-8) ** 0.5
        loss1 = self.e_vdw(dist)
        loss2 = self.e_bond(dist)
        loss3 = self.e_angle(x)
        loss4 = self.e_ele(dist)
        loss5 = self.e_oop(x)
        loss6 = self.e_stretch_bend(x)
        loss7 = self.e_torsion(x)
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        ligand_loss = loss.item()

        # protein part
        if self.protein is not None:
            dist_ligand_protein = (
                ((x[:, None] - self.X[None].detach()) ** 2).sum(-1) + 1e-8
            ) ** 0.5
            if self.protein_prop is not None:
                loss8 = self.e_vdw_ligand_protein(dist_ligand_protein)
                loss9 = self.e_ele_ligand_protein(dist_ligand_protein)
                loss = loss + loss8 + loss9
            else:
                loss10 = torch.relu(self.th - dist_ligand_protein).sum()
                loss = loss + loss10
        return loss, ligand_loss
