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

	'phbilstm.py '
	

### Comparision Experiment
1. Sliding Window Connectivity (SWC) algorithm
	
	' matlab/loworder_net_built.m '

### Data

+ dataset from ADNI rs-fMRI including NC/MCI classification.
+ for the laboratory resources, the dataset is private.

## Publications
The architecture implemented in this repository is described in detail in [a preprint at researchgate](https://www.researchgate.net/profile/Yang-Li-61/publication/336391325_Adaptive_Functional_Connectivity_Network_Using_Parallel_Hierarchical_BiLSTM_for_MCI_Diagnosis/links/5f5dc6aa92851c0789631f76/Adaptive-Functional-Connectivity-Network-Using-Parallel-Hierarchical-BiLSTM-for-MCI-Diagnosis.pdf). If you use this architecture in your research work please cite the paper, with the following bibtex:

```
@inproceedings{jiang2019adaptive,
  title={Adaptive Functional Connectivity Network Using Parallel Hierarchical BiLSTM for MCI Diagnosis},
  author={Jiang, Yiqiao and Huang, Huifang and Liu, Jingyu and Wee, Chong-Yaw and Li, Yang},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={507--515},
  year={2019},
  organization={Springer}
}
``` 