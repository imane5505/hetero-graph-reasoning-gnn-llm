from data_processing.load_hetionet import load_hetionet

data = load_hetionet()

print(data)

from data_processing.filter_graph import filter_hetionet

filtered = filter_hetionet(data)
print("\nFiltered edge types:", filtered.edge_types)
for etype in filtered.edge_types:
    print(
        etype,
        filtered[etype].edge_index.shape
    )