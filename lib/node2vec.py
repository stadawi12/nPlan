from torch_geometric.nn import Node2Vec
import torch
import json
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

def train(model, loader, optimiser):
    """This function is a single loop of node2vec training, 
    we wrap over this function to learn for more epochs."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimiser.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def node2vec(path_data: str, input_data: dict):
    """This function learns an embedding for each graph in a given
    dataset specified in the inputs file.
    It uses the node2vec model to learn an embedding of a graph
    in an unsupervised way.

    Parameters
    ----------
    path_data : str
        path to the directory containing all data
    input_data : dict
        a dictionary containing hyperparameters for training
    """

    import os
    from graph import Graphs

    WL  = input_data['walk_length']
    CS  = input_data['context_size']
    WPN = input_data['walks_per_node']
    NNS = input_data['num_negative_samples']
    P   = input_data['p']
    Q   = input_data['q']
    BS  = input_data['batch_size']
    LR  = input_data['lrN2V']
    NE  = input_data['n_epochsN2V']
    NG  = input_data['num_graphs']
    D   = input_data['datasetN2V']
    SF  = input_data['save_features']

    g = Graphs(path_data, D)

    if NG != None:
        num_graphs = NG
    else:
        num_graphs = len(g)

    for idx in range(num_graphs):

        edge_index = g.get_edges(idx)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Node2Vec(edge_index, embedding_dim=50, 
                walk_length=WL,
                context_size=CS, 
                walks_per_node=WPN, 
                num_negative_samples=NNS, 
                p=P, q=Q, 
                sparse=True).to(device)
        print(model.adj)

        loader = model.loader(batch_size=BS, shuffle=False, 
                num_workers=0)

        optimiser = torch.optim.SparseAdam(list(model.parameters()), 
                lr=LR)

        for epoch in range(1,NE+1):
            loss = train(model, loader, optimiser)
            print(f"Graph_id: {idx+1}, Epoch {epoch}, Loss: {loss:.4f}")

        # if save_features is true then save features
        if SF:

            z = model()
            print(z.shape)

            path_to_dir = os.path.join(path_data, 'features', D)

            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)

            filename = 'id_' + str(idx) + '.pt'

            path_full = os.path.join(path_to_dir, filename)

            torch.save(z.detach().cpu(), path_full)
            
            print(f"Successfully saved features to {path_full}")

        model.adj = None

if __name__ == '__main__':
    from Inputs import Read_Input

    path_inputs = '../inputs.yaml'
    input_data = Read_Input(path_inputs)

    node2vec('../data', input_data)
