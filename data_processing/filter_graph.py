import torch
from torch_geometric.data import HeteroData

KEEP_EDGE_TYPES = {
    ('Compound', 'CtD', 'Disease'),
    ('Compound', 'CbG', 'Gene'),
    ('Compound', 'CuG', 'Gene'),
    ('Compound', 'CdG', 'Gene'),
    ('Gene', 'GpPW', 'Pathway'),
    ('Gene', 'GpBP', 'Biological Process'),
    ('Gene', 'GpMF', 'Molecular Function'),
    ('Disease', 'DaG', 'Gene'),
    ('Disease', 'DuG', 'Gene'),
}

def filter_hetionet(data: HeteroData) -> HeteroData:
    new_data = HeteroData()

    # Copier les nodes (features inchangées)
    for ntype in data.node_types:
        new_data[ntype].x = data[ntype].x

    # Copier uniquement les arêtes utiles
    for etype in data.edge_types:
        if etype in KEEP_EDGE_TYPES:
            new_data[etype].edge_index = data[etype].edge_index

    return new_data
