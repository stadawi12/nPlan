# Introduction

This is an assignment given by nPlan as part of a hiring process. The
task is to train on the provided data.

# Project Overview

Having studied the data in combination with the paper on proteins
by Zitnik & Leskovec, 2017 I have decided to implement a similar
procedure as found in their paper. I found that the data is 
similar enough for such implementation, although there are vast
differences between our data. Nevertheless, I thought reproducing
their implementation might teach me about working with graph
data and about graph neural networks as it is something I have
never been exposed to before.

In their paper, Zitnik & Leskovec have a number of different graphs
for different tissues found in the human body. The nodes in their
graphs represent protein and edges represent how the protein are
connected to each other. In each tissue the proteins have different
functions so that each node (protein) in a graph has a set of labels
associated with it. The difference between the data I have been 
provided and the data from the paper is that their edges can
span multiple graphs, a node from one graph (tissue) can be 
connected to another graph, this is not the case in my data.
Where each graph is distinct and no nodes in one graph are connected
to nodes in another graph.
In the paper, Zitnik & Leskovec use the complex networks of protein
in each tissue to try and infer the functions of proteins in that
tissue. 
They do that by learning the graph representation of each tissue
by embedding each node into a d-dimensional feature space
using advanced machine learning methods. Once obtaining a representation
of a tissue in the feature space, they perform two classification
type tasks. One is a non-transferrable classification task, where
they train a model on features from a single tissue to try and 
predict protein functions (labels) of proteins in the same tissue.
They choose 90% of features in that tissue as training samples and 
the remaining 10% as test samples. The second classification task
is slightly more difficult, where they try to learn functions
of proteins from x number of tissues, to then try and predict 
functions of proteins from a tissue that has never been seen before
by the model.

In my project, I focus on the non-transferable classification task,
where given the graphs, I first learn an embedding of the graph
nodes using node2vec for each individual graph, such that I have
a representation of a given graph in the feature space. 
Then I split the embeded features and labels into 90% training set 
and 10% test set for a classification task, where I try to predict
the labels given the features for each graph seperately. There are
24 graphs in the data I have been provided, for each graph I am
able to create an embedding of that graph into a d-dimensional feature
space (d=50) and, given that I have been provided labels corresponding
to the nodes of the graphs, I am able to run a classifier neural network
to see if I can predict the labels of unseen nodes.

For the classification task I use four different models to see how they
compare and for all of them I use the same loss function,
Binary Cross Entropy Loss (BCELoss). I use this loss because the
labels are k-hot vectors. For the models, I have built simple
linear models with different depths of linear layers with
ReLU activation functions in the hidden layers and a Sigmoid
activation function at the output layer such that my outputs
lie in the range 0 <= out <= 1. As my classifiers are not
performing very well I didn't see the point in building more 
complicated models such as ConvNets or ResNets as I didn't 
believe they would increase performance, although I am perfectly
capable of implementing these models, as I have done so in the past.
Therefore, I stuck to the simple linear models. Had I seen promise
in the linear models I would have moved on to something more 
modern.

To summarise, there are two steps in my project, one is
an embedding task where I learn to represent a single graph in
a d-dimensional feature space and second is a classification task
where I use the learnt features of the graph to predict labels
associated with those features.


## How to get started

To get the code running follow these steps: 

1) download my repository from github into an empty directory.
Using `$ git clone`.

2) Create a fresh python environment (I have used Python version
3.9.5 for my project, would recommend using same python version) using 
`$ python3.9 -m venv .venv` then go ahead and source the new environment
using `$ source .venv/bin/activate`

3) First install torch using `$ pip install torch`

4) After torch is installed correctly, you will need to download
`torch_geometric` which cannot be downloaded using `pip
install` in a straightforward manner.
To download the necessary `torch_geometric` packages, you will
need to visit the
[pytorch geometric installation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
There you should follow steps 1, 2, 3, and in step 4 only install
`torch_cluster` as `torch-spline-conv` is not needed for my application.
Once you have installed all the necessary torch packages, run
`$ pip install -r requirements.txt`. You should now have all
the requirements installed.

5) Once all the requirements are installed, go to the `tests` directory
using `$ cd tests` and run the `Makefile` by writing
`$ make` into the console, this will run all the
unittests I have created making sure that the code works as intended.
If the tests have passed, the code most likely works as intended and 
should be functional on your machine from now onwards.

## How my scripts work

My project has two main actions that can be performed, `embed` and
`classify`. They can be instantiated by running the `main.py` file
and specifying the `-a` flag which stands for action. For example,
to perform an embedding of a graph into a feature space 
we would run the following command `$ python main.py -a embed`, this
would tell `main.py` to train a model on the data provided in the 
`data` folder and to use hyperparameters specified in the `inputs.yaml` 
file. The second action is `classify` which uses the learnt features
of the graph nodes to try and predict the labels of each node.
We can run `$ python main.py -a classify` which will train on features
of a graph specified in the `inputs.yaml` file.
Therefore, `main.py` is the file that executes actions and 
`inputs.yaml` provides instructions for training.

### main.py file

This is the main file of the training project, you can run 
`main.py` using python, you will need to specify a flag, either
`-a embed` or `-a classify`.

### inputs.yaml file

I would recommend getting familiar with this file as it is the 
control centre for all training. It contains hyperparameters for
the node2vec embedding task and also for training a classification
model. In this file you can specify different parameters for training,
like, which `dataset` to use for training `train`, `test` or
`valid`. The number of epochs to train for, the learning rate,
random walk length for node2vec and many more. All these can be
played with after saving the `inputs.yaml` file we can run 
either the embed action or classify action with those parameters.
I find this to be a good way of organising and keeping track of
hyperparameters during training experimentation.

### embed flag

This action (`python main.py -a embed`) will embed a specified 
graph into a 
feature space, and if we set the option `save_features` to `True`,
after `n_epochs` the embedded features will be saved to a file
in the data directory and can be found in 
`data/features/<dataset>/id_<graph_id>.pt`. 
We can choose if we either want to embed a single graph by specifying
`graph_idN2V` or we can embed all graph for a given dataset
by setting `graph_idN2v` to `null`.

### classify flag

The classify flag (`python main.py -a classify`) will take the 
specified features of one of the 24 graphs
(specified in the `inputs.yaml` file) and
the corresponding labels and run a classification task using
the model specified. It will run a training session on 90% of
the features for that graph for `n_epochs` and then try to predict 
the classes of the test set (10% of unseen features). So if we
have 1000 features in a graph, the classifier would train on 900
features and labels and then try to predict 100 unseen labels from
100 unseen features.
We can track the progress of training using tensorboard by setting
`record_run` to `True`. We can view the diagnostics of a training
session using the tensorboard, for this you need to have tensorboard
installed on your machine. Each training run will be saved to a 
directory named `runs`. 

## Observations and Assumptions

I have observed that the embedding task of my project has
worked well. I have checked the similarity of certain nodes
in the feature space for each graph embedding using the cosine
distance to check if two nodes in the same neighbourhood will
have a high similarity in the feature space. My results show
that it is the case which means that node2vec is able to learn
the representation of the graph in the feature space to some
degree. 
The classification task on the other hand has not been successful,
It is only able to predict at most 5% of unseen labels. This
is most likely due to false assumptions made about the data.

I have assumed homophily, that similar nodes will be
classified similarly. Or nodes in similar environments will
have similar labels.

For the classification task I have assumed that the feature
data is independent and identically distributed.

Both assumptions are a very big stretch for the data I have 
been provided. Nevertheless, I have learnt a lot about
graph neural networks, embedding spaces, I have also
researched word embedding methods and sentence embedding methods
using machine learning, models like wrod2vec and doc2vec.

## Final comments

For the embedding, as it is not a model such as word2vec, having
a pretrained model doesn't make a lot of sense to me. My main goal
is to use node2vec to obtain feature vector representation of
a graph rather than
generating a model that is able to embed nodes. If I had 
built a doc2vec or 
similar it would make sense to have a pretrained model that would
take a sentence and be able to embed it into a feature space, 
however, this is not the case in this project. What I do have
however are the pre trained features that have been obtained
using node2vec for every graph, each graph has been learnt using
70 epochs of the node2vec model, the feature 
representations of the graphs can be found in 
`data/features`.

These feature vectors are used for the classification task which can
be ran using `$ python main.py -a classify`.

I haven't provided a pre trained model for the classification
as it did not perform so well. I have made an option to save
a trained model but I haven't made the option to test it on
data as it doesn't perform well. 

Finally, to test that the embedding is working, you can go to
`$ cd lib` and run `$ python test.py` this will select
the feature vectors which have a high similarity with with 
a particular feature and look up if those feature vectors are 
a 1st, 2nd or more than a 2nd neighbour of the selected feature.
Running this test will show that most features that have a 
high similarity with our selected feature are either 1st or 2nd
neighbours in the graph representation G(V,E), this tells us
that the embedding works well to some degree.
