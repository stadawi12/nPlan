# Introduction

This is an assignment given by nPlan as part of a hiring process. The
task is to train on the provided data.

## How to get started

To get the code running follow these steps: 
1) download my repository from github into an empty directory.
Using `$ git clone`.

2) As I do not track the data files themselves, you will need to 
download them into an already existing directory named `data`.

3) After downloading the zipped data file, run
`$ unzip ppi_with_text.zip` followed by
`$ mv ppi/* .`
so that we have all the data files directly in the data directory

4) Now create a fresh python environment (I have used Python version
3.9.5 for my project) using 
`$ python3.9 -m venv .venv` then go ahead and source the new environment
using `$ source .venv/bin/activate`

5) Once that is done, you will be ready to install the necessary python 
packages to run my scripts, to do this, run
`$ pip install -r requirements.txt`,
this may take a while as I am using torch
for my project which alone is ~1GB large, so make sure you have 
enough storage on your machine.

6) Once all the requirements are installed, go to the `tests` directory
using `$ cd tests` and run the `Makefile` by using 
`$ make`, this will run all the
unittests I have created making sure that the code works as intended.
If the tests have passed, the code most likely works as intended and 
should be functional on your machine from now onwards.

## How my scripts work

My project has two main actions that can be performed, `train` and
`test`. They can be instantiated by running the `main.py` file
and specifying the `-a` flag which stands for action. For example,
to train a model 
we would run the following command `$ python main.py -a train`, this
would tell `main.py` to train a model on the data provided in the 
`data` folder and to use parameters specified in the `inputs.yaml` 
file. 

### main.py file

This is the main file of the training project, you can run 
`main.py` using python and you will need to specify a flag, either
`-a train` or `-a test`.

### train flag

This action (`-a train`) will train one of the models I have built.
I have built
5 different but simple models to see which one would perform best and 
to try 
different approaches. To choose a model to train, you will have to 
go into the `inputs.yaml` file, this file contains all the
parameters for training. There, you can choose the number of 
epochs to train for by changing the `n_epochs` variable, you can
set an initial learning rate `lr` and many more. In the 
`inputs.yaml` file you can also specify the model you want to train,
you will see that there are five models to choose from
("linear", "smallLinear", "linRes", "linResBN", "resNet"). They are
all unique models, adding more models is straightforward.
You can go ahead and play with the different variables in the 
`inputs.yaml` file. When starting a training session using 
`$ python main.py -a train`, `main.py` will read the `inputs.yaml`
file and set the specified training parameters. 

### Recording a training session

If we specify 
`record_run : True` the training parameters and diagnostic data
will be saved using tensorboard. Tensorboard is a tool for 
visualising machine learning experiments, it can record how
the loss, validation loss or accuracy evolve during training.
It records the model used for training (in the form of a graph).
Tensorboard also records the hyperparameters used for the training
session, so that you do not lose track of which training session
yielded best results, and many many more. Tensorboard 
is a good tool for 
comparing different 
machine learning experiments. 

By setting `record_run : False` 
the run will not be 
recorded to tensorboard, this is useful if you are debugging and 
do not want to clutter your `runs` directory, which is a directory
where all the runs are saved, and is a directory where tensorboard 
will store the diagnostic data of a run. To view the runs using
the tensorboard API, you will need to have tensorboard installed 
on your system (...). 
To view the runs, use
`$ tensorboard --logdir=runs` from the same directory the `runs`
folder is stored and click the second link which will take you
to the url containing the runs and the diagnostic data of each
training session you have recorded. 

Tensorboard is not a perfect
tool for storing training data but it is something I wanted to 
try out in this project as I have not used it before, I found it
to be quite useful for this project and have a feeling I will
use it for future machine learning projects.

### Saving a trained model

Another option in the `inputs.yaml` file is 
`save_model` you can set that to either `True` or `False` 
depending on whether you would like to save the trained model. 
Again,
not saving a model on every training run is useful for not 
cluttering you memory with badly trained models, once you 
settle on a good choice of hyper-parameters, you can then set
`save_model` to `True` and save your trained model. The 
trained model will be saved to the `\models` directory, where a 
naming convention is specified inside `\models`. 

This should be enough to get started with training a model.
You can have a play around with the hyper-parameters and once
you have agreed on a set of training parameters, save and
exit the `inputs.yaml` file and run 
`python main.py -a train`, this will start a training session.
You will notice that you can also specify the `device` to use
for your training, if you are fortunate enough to have an
alright GPU
on your machine, this will reduce the cost of training 
significantly (compared to training on a `cpu`), simply set
`device : 'cuda:0'` to use a GPU for training.

### test flag

The `-a test` flag is for testing a pre-trained model, so once
we have ran `python main -a train` with the parameter 
`save_model : True` then we should have a 
trained model in the `models` directory. To test the model on
some data, we need to specify; the model we want to test, the
data we want to pass to the model and we need to provide labels
for the data to see how many of our model's predictions are
correct. We can specify all these in the `Test_inputs.yaml`
file. This is a file that `main.py` reads when we run the
`-a test` flag. The parameters in `Test_inputs.yaml` file
should be self explanatory. Once we have the right parameters
set in our `Test_inputs.yaml` file, go ahead and save those
parameters and exit the file, then run: 
`python main.py -a test`, this will start testing the model 
specified, against the data and labels provided. A score will
be given (number of correct predictions / total number of 
examples provided). The predictions will also be stored in 
`\predictions\modelname\dataFileName.npy` 
as a `.npy` file. 

## Assumptions

1) My first assumption is that I should train a neural network 
on the features and labels provided in the assignment. I did
this as I was not convinced that my task was to build a graph
neural network, as upon researching about graph neural networks
I realised that it might take me more than 2 weeks to implement
a graph neural network on the data provided given that I have 
never worked with graph data. For that reason, I have settled on
training a machine learning model to predict the labels provided
in the dataset given the features.

2) I have assumed that the features data is well balanced and 
I have not performed any preprocessing steps on the data.

3) I have assumed that the features data is not locally
dependent, i.e. that feature 4 and 5 in a single example 
are independent of each other.
