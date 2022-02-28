import torch
import pandas as pd
import json

with open('../data/train_graph.json') as f:
    graph = json.load(f)
    links = graph['links']

def get_relation(idx1, idx2):
    neighbours1 = []
    for link in links:
        source = link['source']
        target = link['target']
        if source == idx1:
            neighbours1.append(target)


def cosine(idx1, idx2):
    dot = torch.dot(idx1,idx2)
    norm_1 = torch.norm(idx1)
    norm_2 = torch.norm(idx2)
    distance = dot / (norm_1 * norm_2)
    return distance


features = torch.load('features.pt')
labels = torch.load('labels.pt')

idx1 = features[0]

print(cosine(idx1, features[372]))
