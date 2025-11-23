# chem_graph.py
import os
import torch
from rdkit import Chem

# ---- knobs you can tweak ----
ROLE_ONE_HOT    = True    # add role one-hot to node features
EDGE_ONE_HOT    = True    # one-hot edge types: single,double,triple,aromatic
RICH_NODE_FEATS = True    # richer atom features instead of just Z

BAD_TOKENS = {"0", "none", "nan", ""}
BAD_SMILES = {
    "[Pd](|OC(C)=O)|OC(C)=O",
    "[Fe]|1|2|3|4|5|6|7|8(|[CH-]9[CH-]|1[CH-]|2[C-]|3([CH-]|49)P(c%10ccccc%10)c%11ccccc%11)|C%12=C|5[C-]|6(C|7=C|8%12)P(c%13ccccc%13)c%14ccccc%14",
}

def norm(s):
    if s is None: return None
    s = str(s).strip()
    if s.lower() in BAD_TOKENS: return None
    if s in BAD_SMILES: return None
    return s

# ---------- atom features ----------
_HYBS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    "OTHER",
]

def atom_feat(a):
    """Tensor length 10:
       [Z/100, degree/5, imp_val/6, (charge+2)/4, aromatic, in_ring,
        hyb_SP, hyb_SP2, hyb_SP3, hyb_OTHER]
    """
    Z = a.GetAtomicNum()
    deg = min(a.GetDegree(), 5)
    impv = min(max(a.GetImplicitValence(), 0), 6)
    chg = min(max(a.GetFormalCharge(), -2), 2)
    aromatic = 1.0 if a.GetIsAromatic() else 0.0
    inring   = 1.0 if a.IsInRing() else 0.0
    hyb = a.GetHybridization()
    idx = _HYBS.index(hyb) if hyb in _HYBS[:-1] else 3
    hyb_oh = [1.0 if i == idx else 0.0 for i in range(4)]
    return torch.tensor([Z/100.0, deg/5.0, impv/6.0, (chg+2)/4.0,
                         aromatic, inring, *hyb_oh], dtype=torch.float)

def smiles_to_graph(smiles: str):
    """RDKit SMILES -> torch_geometric.data.Data with node/edge features."""
    from torch_geometric.data import Data
    s = norm(smiles)
    if s is None: return None
    mol = Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms() == 0: return None

    # node features
    if RICH_NODE_FEATS:
        x = torch.stack([atom_feat(a) for a in mol.GetAtoms()], dim=0)  # [N,10]
    else:
        x = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()],
                         dtype=torch.float).view(-1, 1)

    # edges
    ei, ej, eattr = [], [], []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        if EDGE_ONE_HOT:
            oh = [0.0, 0.0, 0.0, 0.0]
            bt = b.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE:   oh[0] = 1.0
            elif bt == Chem.rdchem.BondType.DOUBLE: oh[1] = 1.0
            elif bt == Chem.rdchem.BondType.TRIPLE: oh[2] = 1.0
            elif bt == Chem.rdchem.BondType.AROMATIC: oh[3] = 1.0
            else: oh[0] = 1.0
            eattr += [oh, oh]
        else:
            bt = b.GetBondType()
            w = 1.0
            if bt == Chem.rdchem.BondType.DOUBLE: w = 2.0
            elif bt == Chem.rdchem.BondType.TRIPLE: w = 3.0
            elif bt == Chem.rdchem.BondType.AROMATIC: w = 1.5
            eattr += [[w], [w]]
        ei += [i, j]; ej += [j, i]

    if not ei:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 4 if EDGE_ONE_HOT else 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([ei, ej], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def union_graphs_with_role(graphs, roles, num_roles: int):
    """Disjoint union; appends role features to node features."""
    from torch_geometric.data import Data
    xs, eis, eattrs, off = [], [], [], 0
    for g, r in zip(graphs, roles):
        if g is None: continue
        base_x = g.x
        if ROLE_ONE_HOT:
            role = torch.zeros((g.num_nodes, num_roles), dtype=base_x.dtype)
            role[:, r] = 1.0
            x_aug = torch.cat([base_x, role], dim=1)
        else:
            x_aug = torch.cat([base_x, torch.full((g.num_nodes, 1), float(r), dtype=base_x.dtype)], dim=1)
        xs.append(x_aug)
        if g.edge_index.numel() > 0:
            eis.append(g.edge_index + off)
            ea = g.edge_attr
            if ea.dim() == 1: ea = ea.view(-1, 1)
            eattrs.append(ea)
        off += g.num_nodes

    if not xs:
        return None
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.empty((2, 0), dtype=torch.long)
    ea_dim = 4 if EDGE_ONE_HOT else 1
    edge_attr = torch.cat(eattrs, dim=0) if eattrs else torch.empty((0, ea_dim), dtype=x.dtype)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------- convenience: feature dims for your train script ----------
def feature_dims(num_roles: int):
    node_base = 10 if RICH_NODE_FEATS else 1
    node_in = node_base + (num_roles if ROLE_ONE_HOT else 1)
    edge_in = 4 if EDGE_ONE_HOT else 1
    return node_in, edge_in

# ---------- NEW: dataframe helper used by prep_graphs.py ----------
def graphs_from_dataframe(df, smiles_cols=None, save_dir="mol_graphs",
                          keep_objects=True, verbose=True):
    """
    Builds/saves unique graphs for each SMILES column and returns a new DF:
      - adds '<col>_Graph' (PyG Data object, optional)
      - adds '<col>_GraphPath' (path to saved .pt or None)
    """
    import pandas as pd
    import torch

    if smiles_cols is None:
        smiles_cols = [c for c in df.columns if c.endswith("_SMILES")]

    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
    stats = {}

    for col in smiles_cols:
        series = df[col].astype(str) if col in df else None
        if series is None:
            if verbose: print(f"[{col}] column not found; skipping")
            continue

        uniq = pd.Series(series.dropna().map(norm)).dropna().unique().tolist()
        cache = {}
        ok = failed = 0
        for s in uniq:
            g = smiles_to_graph(s)
            cache[s] = g
            if g is None: failed += 1
            else: ok += 1

        if verbose:
            print(f"[chem_graphs] uniques={len(uniq)} | ok={ok} | failed={failed}")
            print(f"[{col}] saved={ok} failed={failed}")
            if failed:
                bad = [s for s in uniq if cache[s] is None]
                print("  examples:", bad[:1])

        # save to disk + build path map
        path_map = {}
        for s, g in cache.items():
            if g is None:
                path_map[s] = None
            else:
                fn = f"{col}__{abs(hash(s)) & 0xFFFFFFFF:08x}.pt"
                torch.save(g, os.path.join(save_dir, fn))
                path_map[s] = os.path.join(save_dir, fn)

        df[f"{col}_GraphPath"] = series.map(lambda s: path_map.get(norm(s), None))
        if keep_objects:
            df[f"{col}_Graph"] = series.map(lambda s: cache.get(norm(s), None))

        stats[col] = {"uniques": len(uniq), "ok": ok, "failed": failed}

    return df, stats

