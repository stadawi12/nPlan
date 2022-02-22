# Introduction

This is an assignment given by nPlan as part of a hiring process. The
task here is to train a neural network on the provided features and
labels. 

The data consists of roughly 45,000 training examples, each example 
consists of 50 features of data type float. I am also given 45,000 
labels which are k-hot vectors of size 121, that is each label is a 
vector
of 1's and 0's, of size 121. I am also given a validation dataset and a
testing dataset for testing the accuracy of the neural network during
training, and also for validating that the network is not
overfitting. The test data contains around 5,500 feature/label pairs
and validation
dataset contains around 6,500 feature/label pairs.

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
train a model 
we would run the following command `$ python main.py -a train`, this
would tell `main.py` to train a model on the data provided in the 
`data` folder and to use parameters specified in the `inputs.yaml` 
file. 

### main.py file

### inputs.py file
Inside the `inputs.yaml` file you will find a lot 
of parameters that you can tweak for training, you will find things
like `n_epochs` which is the number of epochs you want your training
to run for or `lr` which is the initial learning rate for your
training, etc.. There are many parameters that you can play with 
to within reason, it is the control centre for training models. 
Using the inputs file, you can choose the `model` you want to train
from one of the listed models specified in the comment above it.
There is only one `loss` function implemented in this project but I 
have allowed the project to be scalable and adding a different
loss function would not be difficult.

### train

### test
