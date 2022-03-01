import torch
import pandas as pd
import json
from graph import Graphs

def get_relation(idx1, idx2):
    neighbours1 = []
    for link in links:
        source = link['source'] 
        target = link['target']
        if source == idx1:
            neighbours1.append(target)

    if idx2 in neighbours1:
        return '1st'

    else:
        neighbours2 = []
        for neighbour in neighbours1:
            for link in links:
                source = link['source']
                target = link['target']
                if source == neighbour:
                    neighbours2.append(target)
        if idx2 in neighbours2:
            return '2nd'
        else:
            return '>2nd'

def cosine(node1, node2):
    dot = torch.dot(node1,node2)
    norm_1 = torch.norm(node1)
    norm_2 = torch.norm(node2)
    distance = dot / (norm_1 * norm_2)
    return distance


idx1 = 0
data = Graphs('../data', 'train')
shift = data.get_shift(idx1)
features = data.get_features(idx1)

with open('../data/train_graph.json') as f:
    graph = json.load(f)
    links = graph['links']
    count_1st = 0
    for link in links:
        source = link['source'] 
        target = link['target']
        if source == idx1+shift:
            count_1st += 1
    print(f'Number of 1st neighbours: {count_1st}')


node1 = features[idx1]
for idx2, node2 in enumerate(features):
    similarity = cosine(node1, node2)
    if similarity > 0.5:
        print(idx2, similarity, 
                get_relation(idx1 + shift, idx2 + shift))
