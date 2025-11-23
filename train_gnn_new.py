#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, random
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from edge_aware_gnn import EdgeAwareGNN  # keep your file

# ================== Toggles & Base Config ==================
STANDARDIZE_Y   = True
USE_SCHEDULER   = True
USE_BN          = True
SAVE_VAL_PREDS  = True

# More stable defaults vs your last run
CSV_PATH   = "dataset_6_chloroquinoline.csv"
TARGET_COL = "Product_Yield_PCT_Area_UV"

SMILES_COLS = [
    "Reactant_1_SMILES",
    "Reactant_2_Name_SMILES",
    "Catalyst_1_Short_Hand_SMILES",
    "Ligand_Short_Hand_SMILES",
    "Reagent_1_Short_Hand_SMILES",
    "Solvent_1_Short_Hand_SMILES",
]

# Feature switches
ROLE_ONE_HOT     = True    # one-hot roles (better than numeric)
EDGE_ONE_HOT     = True    # one-hot bonds (S/D/T/A) instead of 1/2/3/1.5
RICH_NODE_FEATS  = True    # add valence/degree/charge/aromatic/ring/hybridization

# Training hyperparams (safer on CPU)
BATCH_SIZE = 64
EPOCHS     = 200
LR         = 1e-4
HIDDEN     = 256
LAYERS     = 3
DROPOUT    = 0.1
VAL_SPLIT  = 0.2
SEED       = 42
WD         = 1e-4
CLIP_NORM  = 1.0
WEIGHT_HIGH_YIELD = False  # start False; you can try True after you stabilize
# Training hyperparams (safer on CPU)
#BATCH_SIZE = 128
#EPOCHS     = 150
#LR         = 3e-4
#HIDDEN     = 256
#LAYERS     = 3
#DROPOUT    = 0.1
#VAL_SPLIT  = 0.2
#SEED       = 42
#WD         = 5e-4
#CLIP_NORM  = 1.0
#WEIGHT_HIGH_YIELD = False  # start False; you can try True after you stabilize

# ================== Misc ==================
for lvl in ("rdApp.error", "rdApp.warning", "rdApp.info", "rdApp.debug"):
    RDLogger.DisableLog(lvl)

BAD_TOKENS = {"0", "none", "nan", ""}
BAD_SMILES = {
    "[Pd](|OC(C)=O)|OC(C)=O",
    "[Fe]|1|2|3|4|5|6|7|8(|[CH-]9[CH-]|1[CH-]|2[C-]|3([CH-]|49)P(c%10ccccc%10)c%11ccccc%11)|C%12=C|5[C-]|6(C|7=C|8%12)P(c%13ccccc%13)c%14ccccc%14",
}

def set_seed(s=SEED):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def norm(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if s.lower() in BAD_TOKENS: return None
    if s in BAD_SMILES: return None
    return s

# ================== Feature Builders ==================
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    "OTHER",
]

def atom_feat(a):
    """Rich per-atom features (no role here).
       Returns tensor length 10:
       [Z/100, degree/5, imp_val/6, (charge+2)/4, aromatic, in_ring,
        hyb_SP, hyb_SP2, hyb_SP3, hyb_OTHER]
    """
    Z = a.GetAtomicNum()
    deg = min(a.GetDegree(), 5)
    impv = min(max(a.GetImplicitValence(), 0), 6)
    chg = a.GetFormalCharge()
    chg = min(max(chg, -2), 2)
    aromatic = 1.0 if a.GetIsAromatic() else 0.0
    inring = 1.0 if a.IsInRing() else 0.0
    hyb = a.GetHybridization()
    hyb_idx = 3  # OTHER
    if hyb in HYB_LIST[:-1]:
        hyb_idx = HYB_LIST.index(hyb)
    hyb_onehot = [1.0 if i==hyb_idx else 0.0 for i in range(4)]
    return torch.tensor([
        Z/100.0,
        deg/5.0,
        impv/6.0,
        (chg+2)/4.0,
        aromatic,
        inring,
        *hyb_onehot
    ], dtype=torch.float)

def smiles_to_graph(smiles: str):
    """RDKit SMILES -> PyG Data with:
       x: rich atom feats (len=10) if RICH_NODE_FEATS else [Z]
       edge_attr: one-hot 4 if EDGE_ONE_HOT else [bond_weight]
    """
    s = norm(smiles)
    if s is None: return None
    mol = Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms() == 0: return None

    if RICH_NODE_FEATS:
        x = torch.stack([atom_feat(a) for a in mol.GetAtoms()], dim=0)  # [N,10]
    else:
        x = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.float).view(-1,1)  # [N,1]

    edges_i, edges_j = [], []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        bt = b.GetBondType()
        if EDGE_ONE_HOT:
            # single, double, triple, aromatic
            one = [0.0,0.0,0.0,0.0]
            if bt == Chem.rdchem.BondType.SINGLE:   one[0]=1.0
            elif bt == Chem.rdchem.BondType.DOUBLE: one[1]=1.0
            elif bt == Chem.rdchem.BondType.TRIPLE: one[2]=1.0
            elif bt == Chem.rdchem.BondType.AROMATIC: one[3]=1.0
            else: one[0]=1.0
            edge_attr += [one, one]
        else:
            if bt == Chem.rdchem.BondType.SINGLE:   w = 1.0
            elif bt == Chem.rdchem.BondType.DOUBLE: w = 2.0
            elif bt == Chem.rdchem.BondType.TRIPLE: w = 3.0
            elif bt == Chem.rdchem.BondType.AROMATIC: w = 1.5
            else: w = 1.0
            edge_attr += [[w],[w]]
        edges_i += [i, j]; edges_j += [j, i]

    if len(edges_i) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        ea_dim = 4 if EDGE_ONE_HOT else 1
        edge_attr = torch.empty((0, ea_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor([edges_i, edges_j], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def union_graphs_with_role(graphs, roles, num_roles):
    """
    Disjoint union. If ROLE_ONE_HOT, append role one-hot to each node.
    Node dim = base_dim + (num_roles if ROLE_ONE_HOT else 1)
    Edge dim = 4 (one-hot) or 1 (weight)
    """
    xs, eis, eattrs, offset = [], [], [], 0
    for g, r in zip(graphs, roles):
        if g is None: continue
        base_x = g.x
        if ROLE_ONE_HOT:
            role_vec = torch.zeros((g.num_nodes, num_roles), dtype=base_x.dtype)
            role_vec[:, r] = 1.0
            x_aug = torch.cat([base_x, role_vec], dim=1)
        else:
            role_scalar = torch.full((g.num_nodes, 1), float(r), dtype=base_x.dtype)
            x_aug = torch.cat([base_x, role_scalar], dim=1)
        xs.append(x_aug)

        if g.edge_index.numel() > 0:
            eis.append(g.edge_index + offset)
            ea = g.edge_attr
            if ea.dim() == 1: ea = ea.view(-1,1)
            eattrs.append(ea)
        offset += g.num_nodes

    if not xs:
        return None

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.empty((2,0), dtype=torch.long)
    edge_attr  = torch.cat(eattrs, dim=0) if eattrs else torch.empty((0, (4 if EDGE_ONE_HOT else 1)), dtype=x.dtype)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ================== Dataset ==================
class ReactionRowDataset(Dataset):
    def __init__(self, df, smiles_cols, target_col):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.smiles_cols = smiles_cols
        self.target_col  = target_col
        self.num_roles   = len(smiles_cols)
        self.cache = {}

        self.valid_idx = []
        for i in range(len(self.df)):
            row = self.df.loc[i]
            try:
                y_val = float(row.get(self.target_col, None))
                if pd.isna(y_val): continue
            except Exception:
                continue

            ok = False
            for col in self.smiles_cols:
                if col not in self.df.columns: continue
                s = norm(row[col])
                if s is None: continue
                g = self.cache.get(s)
                if g is None and s not in self.cache:
                    g = smiles_to_graph(s)
                    self.cache[s] = g
                if g is not None:
                    ok = True
                    break
            if ok: self.valid_idx.append(i)

        print(f"Rows total: {len(self.df)} | usable: {len(self.valid_idx)}")

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        row = self.df.loc[i]

        graphs, roles = [], []
        for role_id, col in enumerate(self.smiles_cols):
            if col not in self.df.columns: continue
            s = norm(row[col]); 
            if s is None: continue
            g = self.cache.get(s)
            if g is None and s not in self.cache:
                g = smiles_to_graph(s); self.cache[s] = g
            if g is not None:
                graphs.append(g); roles.append(role_id)

        G = union_graphs_with_role(graphs, roles, num_roles=self.num_roles)
        val = float(row[self.target_col])
        y  = torch.tensor([(val - y_mean) / y_std], dtype=torch.float)

        # Fallback (rare)
        if G is None:
            base_dim = (10 if RICH_NODE_FEATS else 1)
            role_dim = (self.num_roles if ROLE_ONE_HOT else 1)
            ea_dim   = (4 if EDGE_ONE_HOT else 1)
            G = Data(x=torch.zeros((1, base_dim+role_dim), dtype=torch.float),
                     edge_index=torch.empty((2,0), dtype=torch.long),
                     edge_attr=torch.empty((0,ea_dim), dtype=torch.float))
        G.y = y
        return G

# ================== Helpers ==================
def infer_dims(loader, device):
    for b in loader:
        b = b.to(device)
        node_in = int(b.x.size(1))
        edge_in = int(b.edge_attr.size(1)) if getattr(b, "edge_attr", None) is not None else 0
        return node_in, edge_in
    # worst-case defaults
    return (10 if RICH_NODE_FEATS else 1) + (len(SMILES_COLS) if ROLE_ONE_HOT else 1), (4 if EDGE_ONE_HOT else 1)

def unscale(t):  # standardized
    return t * y_std + y_mean

# ================== Train ==================
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(CSV_PATH)
    global y_mean, y_std
    if STANDARDIZE_Y:
        y_mean = float(pd.to_numeric(df[TARGET_COL], errors="coerce").mean())
        y_std  = float(pd.to_numeric(df[TARGET_COL], errors="coerce").std() + 1e-8)
        print(f"[scale] {TARGET_COL} mean={y_mean:.3f}, std={y_std:.3f}")
    else:
        y_mean, y_std = 0.0, 1.0

    use_cols = [c for c in SMILES_COLS if c in df.columns]
    if len(use_cols) == 0:
        raise ValueError("None of the specified SMILES_COLS exist in the CSV.")

    ds_all = ReactionRowDataset(df, use_cols, TARGET_COL)

    n = len(ds_all)
    idx = list(range(n)); random.shuffle(idx)
    v = max(1, int(n * VAL_SPLIT))
    val_idx, train_idx = idx[:v], idx[v:]

    train_ds = torch.utils.data.Subset(ds_all, train_idx)
    val_ds   = torch.utils.data.Subset(ds_all, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    node_in, edge_in = infer_dims(train_loader, device)
    print(f"[dims] node_in={node_in} edge_in={edge_in}")

    model = EdgeAwareGNN(
        node_in=node_in, edge_in=edge_in,
        hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT,
        out_dim=1, use_bn=USE_BN
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    if USE_SCHEDULER:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=6, min_lr=1e-6, verbose=True
        )

    best_val = float("inf")
    patience, bad = 24, 0

    hist = {"train": [], "val": [], "lr": [], "t": []}
    t0 = time.time()

    def run_epoch(loader, train=True):
        if train: model.train()
        else:     model.eval()
        tot, m = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(device)
                if train: opt.zero_grad()
                pred = model(batch).view(-1)
                y_std_tgt = batch.y.view(-1)

                if WEIGHT_HIGH_YIELD:
                    y_real = y_std_tgt * y_std + y_mean
                    thr = torch.quantile(y_real, 0.75)
                    w = torch.where(y_real >= thr, 2.0, 1.0)
                else:
                    w = 1.0

                # robust loss without aggressive weighting
                mse   = F.mse_loss(pred, y_std_tgt, reduction='none')
                huber = F.smooth_l1_loss(pred, y_std_tgt, beta=1.0, reduction='none')
                loss  = (0.5*mse + 0.5*huber) * w
                loss  = loss.mean()

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                    opt.step()

                tot += loss.item() * batch.num_graphs
                m   += batch.num_graphs
        return tot / max(1, m)

    for ep in range(1, EPOCHS+1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader,   False)
        hist["train"].append(tr); hist["val"].append(va)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["t"].append(time.time() - t0)

        if USE_SCHEDULER: sched.step(va)
        print(f"Epoch {ep:03d} | train MSE {tr:.4f} | val MSE {va:.4f} | lr {opt.param_groups[0]['lr']:.2e}")

        if va + 1e-6 < best_val:
            best_val, bad = va, 0
            torch.save(model.state_dict(), "gnn_yield_min.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    torch.save(model.state_dict(), "gnn_yield_min.pt")
    print("Saved model to gnn_yield_min.pt")

    # ---- Plots ----
    os.makedirs("figs", exist_ok=True)

    plt.figure(); plt.plot(hist["train"], label="train MSE"); plt.plot(hist["val"], label="val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE (standardized)"); plt.title("Train/Val MSE vs Epoch")
    plt.legend(); plt.tight_layout(); plt.savefig("figs/loss_curves.png", dpi=150); plt.close()

    plt.figure(); plt.plot(hist["t"], hist["val"])
    plt.xlabel("Time (s)"); plt.ylabel("Val MSE (standardized)"); plt.title("Validation MSE vs Time")
    plt.tight_layout(); plt.savefig("figs/val_mse_vs_time.png", dpi=150); plt.close()

    plt.figure(); plt.plot(hist["lr"])
    plt.xlabel("Epoch"); plt.ylabel("Learning rate"); plt.title("LR schedule")
    plt.tight_layout(); plt.savefig("figs/lr_schedule.png", dpi=150); plt.close()
    print("Saved: figs/loss_curves.png, figs/val_mse_vs_time.png, figs/lr_schedule.png")

    # ---- Final metrics in original units + parity ----
    def collect_preds(loader):
        model.eval(); Ys, Ps = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                Ps.append(model(b).view(-1).cpu())
                Ys.append(b.y.view(-1).cpu())
        if not Ys: return None, None
        return torch.cat(Ys), torch.cat(Ps)

    Y_std, P_std = collect_preds(val_loader)
    if Y_std is not None:
        Y = unscale(Y_std); P = unscale(P_std)
        mse = torch.mean((P - Y)**2).item()
        mae = torch.mean(torch.abs(P - Y)).item()
        ss_res = torch.sum((P - Y)**2)
        ss_tot = torch.sum((Y - torch.mean(Y))**2) + 1e-8
        r2 = 1.0 - (ss_res/ss_tot).item()
        print(f"[VAL] MSE={mse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

        if SAVE_VAL_PREDS:
            pd.DataFrame({"y_true": Y.numpy(), "y_pred": P.numpy()}).to_csv("val_preds.csv", index=False)
            print("Wrote val_preds.csv")

        err = P - Y
        rmse = torch.sqrt((err**2).mean()).item()
        print(f"[VAL] RMSE={rmse:.2f}")
        
        abs_err = err.abs()
        N = Y.numel()
        idx = torch.randperm(N)[:min(10, N)]
        print("\nSample of 10 validation cases (original units):")
        print(f"{'idx':>4} {'y_true':>9} {'y_pred':>9} {'err':>9} {'|err|':>9}")
        for i in idx.tolist():
            print(f"{i:4d} {Y[i].item():9.2f} {P[i].item():9.2f} {err[i].item():9.2f} {abs_err[i].item():9.2f}")
        rmse = torch.sqrt((err**2).mean()).item()

        import numpy as np
        y_true = Y.numpy(); y_pred = P.numpy()
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        plt.figure(); plt.scatter(y_true, y_pred, s=14, alpha=0.7)
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("True Yield (PCT_Area_UV)"); plt.ylabel("Predicted Yield")
        plt.title("Parity Plot: Predicted vs True"); plt.tight_layout()
        plt.savefig("figs/parity_val.png", dpi=150); plt.close()
        print("Saved: figs/parity_val.png")
        print(f"[VAL] Acc_exact={(P.eq(Y)).float().mean().item()*100:.1f}%")

    else:
        print("Validation loader is empty; nothing to show.")

if __name__ == "__main__":
    main()

