## PH-BiLSTM

### Procedure Code Introduction
1. Detect the topology of efficient connectivity networks via group-constrained structure detection algorithm;

2. Apply an adaptive Kalman Filter algorithm to recursively estimate the efficient connectivity strength; 

3. Split the dataset into train data and test data, extract topological features from adaptive FC networks with the optimal parameters, train a PH-BiLSTM model and test it with the test data. 


### Run Matlab & Python
1. Implement GroupLasso algorithm

	' matlab/GroupLasso.m '  
	
2. Implement adaptive dFC via Kalman filter algorithm by RARX matlab toolbox

	' matlab/rarx_kf.m '
	
	' matlab/get_kalmanDim.m     % convert and squeeze matrix '

3. Implement PH-BiLSTM by Tensorflow and Keras deep learning model

	' phbilstm.py '
	

### Comparision Experiment
1. Sliding Window Connectivity (SWC) algorithm
	
	' matlab/loworder_net_built.m '

### Data

+ dataset from ADNI rs-fMRI including NC/MCI classification.
+ for the laboratory resources, the dataset is private.