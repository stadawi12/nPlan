This is an assignment given by nPlan as part of a hiring process. The
task here is to train a neural network on the provided features and
labels. 

The data consists of roughly 45,000 training examples, each example 
consists of 50 features of data type float. I am also given 45,000 
labels which are k-hot vectors of size 121, that is each label is a vector
of 1's and 0's, of size 121. I am also given a validation dataset and a
testing dataset for testing the accuracy of the neural network during
training and after, and also for validating that the network is not
overfitting. The test data contains around 5,000 examples and validation
dataset contains around 6,000 examples.

To get the code running, run `$ pip install -r requirements.txt` this
will install all the required python packages needed to run my code.
I have used python version 3.9.5 for this project, it is most likely
that you will need the same version of python to run the code. Some
packages that will be installed on your system might not be necessary to
run the code but were needed to run certain things on my system, for
example, syntax highlightin, apologies for putting those in the
requirements file as it would be too tedious for me to remove some of
those packages as they contain co-dependencies, good news is they
shouldn't take up too much space in any case.
