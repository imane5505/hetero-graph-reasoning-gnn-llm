from torch_geometric.utils import negative_sampling


def prepare_ctd(data):
   
    # Type d'arête cible
    etype = ('Compound', 'CtD', 'Disease')

    # Arêtes positives (labels)
    pos_edge_index = data[etype].edge_index

    # Tailles des ensembles
    num_compounds = data['Compound'].num_nodes
    num_diseases = data['Disease'].num_nodes

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(num_compounds, num_diseases),
        num_neg_samples=pos_edge_index.size(1)
    )

    # ⚠️ NE PAS supprimer CtD du graphe
    return pos_edge_index, neg_edge_index, data
