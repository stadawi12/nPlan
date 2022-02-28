from torch_geometric.nn import Node2Vec
import torch
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimiser.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


g = Graphs('../data', 'train')
for idx in range(len(g))
    x, edge_index, labels = g[idx]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(edge_index, embedding_dim=50, walk_length=20,
            context_size=10, walks_per_node=20, 
            num_negative_samples=2, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=False, num_workers=0)
    optimiser = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    for epoch in range(1,101):
        loss = train()
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    z = model()
    torch.save(z.detach().cpu(), 'features.pt')
    torch.save(labels, 'labels.pt')
