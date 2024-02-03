# FINEST: Stablizing Recommendations by Rank-Preserving Fine-Tuning

Overview
---------------
**FINEST: Stablizing Recommendations by Rank-Preserving Fine-Tuning**
[Sejoon Oh](https://sejoonoh.github.io/), [Berk Ustun](https://www.berkustun.com/), [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/), and [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)  

This repository contains the code and dataset used in the FINEST paper.  
[Link to Dataset](https://drive.google.com/file/d/1AvpAqBQvr0BduHDPVqssM5KIyIO_hHa4/view?usp=sharing)  
**sh Demo.sh** command will execute FINIST on the LSTM model and reddit dataset (standard training for 50 epochs + fine-tuning for 50 epochs).  


Usage
---------------

The detailed execution procedure of **FINEST** is given as follows.

1) Install all required libraries by "pip install -r requirements.txt".
2) "src/main.py" and "src/LSTM.py" include data preparation/perturbation and training/evaluation parts of FINEST on the LSTM recommendation model, respectively.
3) "python src/main.py [arguments]" will execute FINEST on the LSTM model with arguments.

Note that the fine-tuning speed can be significantly improved by optimizing perturbation simulation part.
Please contact the first author (Sejoon Oh) for the code of FINEST for other recommendation models.

Demo
---------------
To run the demo, please follow the following procedure. **FINEST** demo will be executed with the Reddit dataset.

	1. Check permissions of files (if not, use the command "chmod 777 *")
	2. Execute "./demo.sh"
	3. Check "LSTM_stability" for the demo result of FINEST on Reddit dataset

Abstract
---------------
Modern recommender systems may generate significantly different recommendations due to small perturbations in the training data. Changes in the data from one user can alter the recommendations for other unrelated users. We propose a method to stabilize recommender systems against such perturbations. This is a challenging task due to (1) the unavailability of “ideal” ranked recommendation lists; (2) the scalability of optimizing the stability of rank lists containing all items and for all training instances; and (3) the possibility of various noisy perturbations. Our method, FINEST, overcomes these challenges by first obtaining reference rank lists from a given recommendation model and then fine-tuning the model under simulated perturbation scenarios with rank-preserving regularization on sampled items. Our experiments on three real-world datasets demonstrate that FINEST can ensure that recommender models produce stable recommendations under a wide range of different perturbations while preserving next-item prediction accuracy.
