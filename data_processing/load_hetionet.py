import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from pathlib import Path


def parse_node(node_str):
    """
    Exemple :
    'Gene::9021' -> ('Gene', '9021')
    'Biological Process::GO:0071357' -> ('Biological Process', 'GO:0071357')
    """
    node_type, node_id = node_str.split("::", 1)
    return node_type, node_id


def load_hetionet(
    nodes_path="data/raw/hetionet/hetionet-v1.0-nodes.tsv",
    edges_path="data/raw/hetionet/edges.tsv",
):
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)

    # =========================
    # Load TSV files
    # =========================
    nodes_df = pd.read_csv(nodes_path, sep="\t")
    edges_df = pd.read_csv(edges_path, sep="\t")

    # nodes_df columns: id, name, kind
    # edges_df columns: source, metaedge, target

    data = HeteroData()

    # =========================
    # Build node mappings
    # =========================
    node_maps = {}  # {node_type: {full_node_id: index}}

    for node_type in nodes_df["kind"].unique():
        subset = nodes_df[nodes_df["kind"] == node_type]

        # IMPORTANT: IDs complets, ex: "Gene::9021"
        full_ids = subset["id"].astype(str).tolist()

        node_maps[node_type] = {
            full_id: idx for idx, full_id in enumerate(full_ids)
        }

        # Features factices (identity)
        data[node_type].x = torch.eye(len(full_ids))

    # =========================
    # Build edges
    # =========================
    edge_dict = defaultdict(list)

    for _, row in edges_df.iterrows():
        src_full = row["source"]   # ex: Gene::9021
        dst_full = row["target"]   # ex: Disease::DOID:9352
        rel = row["metaedge"]      # ex: CtD, GpBP, DrD, ...

        src_type, _ = parse_node(src_full)
        dst_type, _ = parse_node(dst_full)

        if src_type not in node_maps or dst_type not in node_maps:
            continue

        src_idx = node_maps[src_type].get(src_full)
        dst_idx = node_maps[dst_type].get(dst_full)

        if src_idx is None or dst_idx is None:
            continue

        edge_dict[(src_type, rel, dst_type)].append([src_idx, dst_idx])

    # =========================
    # Convert edges to edge_index
    # =========================
    for (src_type, rel, dst_type), edges in edge_dict.items():
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data[(src_type, rel, dst_type)].edge_index = edge_index

    return data
