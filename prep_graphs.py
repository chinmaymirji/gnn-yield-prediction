# prep_graphs.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
   
import pandas as pd
from chem_graph_2 import graphs_from_dataframe

CSV_PATH = "dataset_6_chloroquinoline.csv"
SAVE_ROOT = "mol_graphs"
SMILES_COLS = [
    "Reactant_1_SMILES",
    "Reactant_2_Name_SMILES",
    "Catalyst_1_Short_Hand_SMILES",
    "Ligand_Short_Hand_SMILES",
    "Reagent_1_Short_Hand_SMILES",
    "Solvent_1_Short_Hand_SMILES",
]

df = pd.read_csv(CSV_PATH)

# role_id encodes which column a node came from (0..len-1)
for rid, col in enumerate(SMILES_COLS):
    if col not in df.columns: 
        print(f"Skip missing column: {col}")
        continue
    out_dir = os.path.join(SAVE_ROOT, col)
    df, graphs_map, failed = graphs_from_dataframe(
        df, smiles_col=col, role_id=rid, save_dir=out_dir
    )
    print(f"[{col}] saved={sum(g is not None for g in graphs_map.values())} failed={len(failed)}")
    if failed:
        print("  examples:", failed[:5])

# Persist both “with objects” and “with paths only”
df.to_pickle("df_with_graphs.pkl")
(df.drop(columns=[c for c in df.columns if c.endswith("_Graph")], errors="ignore")
  .to_csv("df_with_graph_paths.csv", index=False))

print("\nWrote:")
print("  - df_with_graphs.pkl (contains Data objects; good for Python)")
print("  - df_with_graph_paths.csv (no objects; keeps *_GraphPath columns)")

