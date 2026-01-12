import torch
from torch.nn.functional import binary_cross_entropy_with_logits

def train_step(encoder, decoder, data, pos_edge, neg_edge, optimizer):
    encoder.train()
    optimizer.zero_grad()

    z = encoder(data.x_dict, data.edge_index_dict)

    src, dst = pos_edge
    pos_score = decoder(z['Compound'][src], z['Disease'][dst])
    pos_label = torch.ones_like(pos_score)

    src, dst = neg_edge
    neg_score = decoder(z['Compound'][src], z['Disease'][dst])
    neg_label = torch.zeros_like(neg_score)

    loss = binary_cross_entropy_with_logits(
        torch.cat([pos_score, neg_score]),
        torch.cat([pos_label, neg_label])
    )

    loss.backward()
    optimizer.step()

    return loss.item()
