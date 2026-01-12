import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from collections import defaultdict

from data_processing.load_hetionet import load_hetionet
from data_processing.filter_graph import filter_hetionet


# =========================
# Load graph
# =========================
data = load_hetionet()
data = filter_hetionet(data)

# =========================
# Load node metadata
# =========================
NODES_PATH = "data/raw/hetionet/hetionet-v1.0-nodes.tsv"
nodes_df = pd.read_csv(NODES_PATH, sep="\t")

# Build mapping: type -> list of names in index order
type_to_names = {}

for node_type in data.node_types:
    names = nodes_df[nodes_df["kind"] == node_type]["name"].tolist()
    type_to_names[node_type] = names


def node_name(node_type, idx):
    try:
        return type_to_names[node_type][idx]
    except IndexError:
        return f"{node_type}:{idx}"


# =========================
# Build adjacency (same as before)
# =========================
compound_to_genes = defaultdict(set)
disease_to_genes = defaultdict(set)
gene_to_pathways = defaultdict(set)

for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        if src_type == "Compound" and dst_type == "Gene":
            compound_to_genes[s].add(d)
        elif src_type == "Disease" and dst_type == "Gene":
            disease_to_genes[s].add(d)
        elif src_type == "Gene" and dst_type == "Pathway":
            gene_to_pathways[s].add(d)


# =========================
# Extract explanations
# =========================
def extract_explanations(compound_idx, disease_idx, max_genes=5):
    explanations = []

    common_genes = (
        compound_to_genes[compound_idx]
        & disease_to_genes[disease_idx]
    )

    for gene_idx in list(common_genes)[:max_genes]:
        explanations.append([
            ("Compound", compound_idx),
            ("Gene", gene_idx),
            ("Disease", disease_idx),
        ])

        for pathway_idx in gene_to_pathways.get(gene_idx, []):
            explanations.append([
                ("Compound", compound_idx),
                ("Gene", gene_idx),
                ("Pathway", pathway_idx),
                ("Disease", disease_idx),
            ])

    return explanations


# =========================
# USE THE FOUND PAIR
# =========================
compound_idx = 869
disease_idx = 0

paths = extract_explanations(compound_idx, disease_idx)

print(f"\n🧠 Explanatory paths for Compound {compound_idx} and Disease {disease_idx}:\n")

for p in paths[:10]:
    readable = [node_name(t, i) for t, i in p]
    print(" → ".join(readable))
