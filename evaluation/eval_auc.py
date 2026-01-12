import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))



import torch
from sklearn.metrics import roc_auc_score

from data_processing.load_hetionet import load_hetionet
from data_processing.filter_graph import filter_hetionet
from training.prepare_ctd import prepare_ctd
from models.gnn.hgt import HGTEncoder
from models.link_prediction.decoder import DotDecoder


@torch.no_grad()
def evaluate_auc(encoder, decoder, data, pos_edge, neg_edge):
    encoder.eval()

    z = encoder(data.x_dict, data.edge_index_dict)

    # Positive scores
    src, dst = pos_edge
    pos_scores = decoder(
        z['Compound'][src],
        z['Disease'][dst]
    )

    # Negative scores
    src, dst = neg_edge
    neg_scores = decoder(
        z['Compound'][src],
        z['Disease'][dst]
    )

    scores = torch.cat([pos_scores, neg_scores]).cpu()
    labels = torch.cat([
        torch.ones(len(pos_scores)),
        torch.zeros(len(neg_scores))
    ])

    auc = roc_auc_score(labels.numpy(), scores.numpy())
    return auc


# =========================
# Load graph
# =========================
data = load_hetionet()
data = filter_hetionet(data)

# Add self-loops (important)
for ntype in data.node_types:
    n = data[ntype].num_nodes
    idx = torch.arange(n)
    edge_index = idx.unsqueeze(0).repeat(2, 1)
    data[(ntype, 'self_loop', ntype)].edge_index = edge_index

# Prepare CtD
pos_edge, neg_edge, data = prepare_ctd(data)

# Load model
encoder = HGTEncoder(hidden_dim=64, data=data)
decoder = DotDecoder()

# ⚠️ IMPORTANT
# Ici, pour l'instant, on évalue un modèle NON entraîné.
# Juste après, on fera l'évaluation du modèle entraîné.

auc = evaluate_auc(encoder, decoder, data, pos_edge, neg_edge)
print(f"AUC (random init model): {auc:.4f}")
