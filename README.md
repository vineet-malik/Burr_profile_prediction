# Burr_profile_prediction
Burr-profile prediction in micro drilling of ductile materials using artificial neural network.
Burrs are classified into two profiles: 1. Uniform burr and 2. Petal burr. The input variables are: cutting speed, feed rate, drill diameter, hardness ratio and tensile strength.
The model is designed in tensorflow. Adam optimisation is used. Relu is used as activation function in the hidden layers and sigmoid in the output layer. Variance scaling initializer is used to initialise weights. The model is able to achieve an accuracy of more than 75%. 

Files:
1. burr_profile_prediction.py -> python code for training model with optimal training variables
2. dataset_mtp_vineet.csv ->  a csv file containg the data used to train the model
3. model_optimisation.ipynb -> Python notebook that is used to find out the optimal training variables(epochs and batch size) for the model
