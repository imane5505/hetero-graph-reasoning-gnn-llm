import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from data_processing.load_hetionet import load_hetionet
from data_processing.filter_graph import filter_hetionet
from training.prepare_ctd import prepare_ctd
from models.gnn.hgt import HGTEncoder
from models.link_prediction.decoder import DotDecoder


@torch.no_grad()
def predict_topk_for_disease(encoder, decoder, data, disease_idx, k=10):
    encoder.eval()
    decoder.eval()

    z = encoder(data.x_dict, data.edge_index_dict)

    comp_emb = z["Compound"]                 # [num_compounds, dim]
    dis_emb = z["Disease"][disease_idx]      # [dim]

    scores = decoder(comp_emb, dis_emb.unsqueeze(0).repeat(comp_emb.size(0), 1))
    topk_scores, topk_idx = torch.topk(scores, k)

    return topk_idx.tolist(), topk_scores.tolist()


# =========================
# Load graph
# =========================
data = load_hetionet()
data = filter_hetionet(data)

# Add self-loops (same as training)
for ntype in data.node_types:
    n = data[ntype].num_nodes
    idx = torch.arange(n)
    edge_index = idx.unsqueeze(0).repeat(2, 1)
    data[(ntype, "self_loop", ntype)].edge_index = edge_index

# Prepare CtD (for consistency)
_, _, data = prepare_ctd(data)

# Load trained model
encoder = HGTEncoder(hidden_dim=64, data=data)
decoder = DotDecoder()

checkpoint = torch.load("trained_model.pt", map_location="cpu")
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])


# =========================
# Choose a disease
# =========================
# ⚠️ Pour l’instant, on prend un index arbitraire
disease_idx = 0   # tu pourras changer

topk_idx, topk_scores = predict_topk_for_disease(
    encoder, decoder, data, disease_idx, k=10
)

print(f"Top-10 predicted drugs for Disease index {disease_idx}:")
for rank, (ci, score) in enumerate(zip(topk_idx, topk_scores), 1):
    print(f"{rank:02d}. Compound #{ci} | score = {score:.4f}")
