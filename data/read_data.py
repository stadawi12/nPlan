import numpy as np
import matplotlib.pyplot as plt 
import json

if __name__ == '__main__':

    # train_feats = np.load('train_feats.npy')
    # train_labels = np.load('train_labels.npy')
    # train_graph_id = np.load('train_graph_id.npy')
    # sorted_ids = list(train_graph_id)
    # sorted_ids.sort()
    # sorted_ids = np.array(sorted_ids)
    # print(np.array_equal(train_graph_id, sorted_ids))
    # print(train_feats.shape)
    # print(train_feats[1])
    # print(train_labels.shape)
    # print(set(train_graph_id))
    # print(train_labels[240])

    # for i, label in enumerate(train_labels):
        # if np.array_equal(label, train_labels[0]):
            # print(i)

    # test_feats = np.load('test_feats.npy')
    # test_labels = np.load('test_labels.npy')
    # test_graph_id = np.load('test_graph_id.npy')
    # print(test_feats.shape)
    # print(test_labels.shape)
    # print(set(test_graph_id))

    valid_feats = np.load('valid_feats.npy')
    # valid_labels = np.load('valid_labels.npy')
    # valid_graph_id = np.load('valid_graph_id.npy')
    print(valid_feats.shape)
    # print(valid_labels.shape)
    # print(valid_graph_id.shape)
    # s = {x: 0 for x in set(train_graph_id)}
    # print(train_graph_id)
    # print(set(valid_graph_id))

    # s = {x: 0 for x in set(train_graph_id)}
    # for el in train_graph_id:
        # s[el] += 1
    # print(s)


    def plot_bars(dataset: str):
        graph_id = np.load(dataset+'_graph_id.npy')
        s = {x: 0 for x in set(graph_id)}
        for el in graph_id:
            s[el] += 1

        ids = s.keys()
        values = s.values()
        plt.bar(ids, values)
        plt.show()

    # plot_bars('train')
    # plot_bars('test')
    # plot_bars('valid')

    with open("train_graph.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        links = jsonObject['links']
        # print(jsonObject.keys())
        # print("Is the graph directed: ", jsonObject['directed'])
        # print("Is it a multigraph: ", jsonObject['multigraph'])
        # print("graph: ", jsonObject['graph'])
        # print("length of nodes: ", len(jsonObject['nodes']))
        # print("Example of a node: ", jsonObject['nodes'][0])
        # print("Length of links: ", len(jsonObject['links']))
        # print("Example of a link: ", jsonObject['links'][0:10])
        # for link in links:
        #     if link['source'] == 1101:
        #         print(link)
        # print("Node 0 is linked to node: ", jsonObject['nodes'][1767])
        # size = len(jsonObject["nodes"])
        # # Adjacency matrix
        # A = np.zeros((size, size))
        # # Fill in adjacency matrix
        # for link in jsonObject["links"]:
        #     row = link["source"]
        #     col = link["target"]
        #     A[row,col] = 1
        # block = 1000
        # print(np.sum(A[0:block,0:block].T - A[0:block,0:block]))

        jsonFile.close()


