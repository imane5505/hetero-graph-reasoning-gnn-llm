import torch

from data_processing.load_hetionet import load_hetionet
from data_processing.filter_graph import filter_hetionet
from training.prepare_ctd import prepare_ctd
from models.gnn.hgt import HGTEncoder
from models.link_prediction.decoder import DotDecoder
from training.train_ctd import train_step


# =========================
# Load & filter graph
# =========================
data = load_hetionet()
data = filter_hetionet(data)


# =========================
# Add self-loops (IMPORTANT)
# =========================
for ntype in data.node_types:
    n = data[ntype].num_nodes
    idx = torch.arange(n)
    edge_index = idx.unsqueeze(0).repeat(2, 1)
    data[(ntype, 'self_loop', ntype)].edge_index = edge_index


# =========================
# Prepare link prediction
# =========================
pos_edge, neg_edge, data = prepare_ctd(data)


# =========================
# Model
# =========================
encoder = HGTEncoder(hidden_dim=64, data=data)
decoder = DotDecoder()

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-3
)


# =========================
# Training loop
# =========================
for epoch in range(1, 21):
    loss = train_step(
        encoder,
        decoder,
        data,
        pos_edge,
        neg_edge,
        optimizer
    )
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")

   


# =========================
# Save model (AFTER loop)
# =========================
torch.save(
    {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
    },
    "trained_model.pt"
)
