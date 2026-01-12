import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from collections import defaultdict

from data_processing.load_hetionet import load_hetionet
from data_processing.filter_graph import filter_hetionet


# =========================
# Load graph
# =========================
data = load_hetionet()
data = filter_hetionet(data)

# Add self-loops (consistency)
for ntype in data.node_types:
    n = data[ntype].num_nodes
    idx = torch.arange(n)
    edge_index = idx.unsqueeze(0).repeat(2, 1)
    data[(ntype, "self_loop", ntype)].edge_index = edge_index


# =========================
# Build adjacency
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
# Path extraction via common genes
# =========================
def extract_explanations(compound_idx, disease_idx, max_genes=10):
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
# Try multiple compounds
# =========================
disease_idx = 0
candidate_compounds = [896, 869, 1425, 214, 581, 317, 1110, 799, 300, 5]

found = False

for compound_idx in candidate_compounds:
    paths = extract_explanations(compound_idx, disease_idx)
    if len(paths) > 0:
        print(f"\n✅ Found explanations for Compound {compound_idx}:")
        for p in paths[:5]:
            print(" → ".join([f"{t}:{i}" for t, i in p]))
        found = True
        break

if not found:
    print("\n❌ No explanatory structures found for Top-10 compounds.")

