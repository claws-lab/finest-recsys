# FINEST: Stablizing Recommendations by Rank-Preserving Fine-Tuning

Overview
---------------
[**FINEST: Stablizing Recommendations by Rank-Preserving Fine-Tuning**](https://arxiv.org/abs/2402.03481)  
[Sejoon Oh](https://sejoonoh.github.io/), [Berk Ustun](https://www.berkustun.com/), [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/), and [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)  

This repository contains the code and dataset used in the FINEST paper.  
[Link to Dataset](https://drive.google.com/file/d/1AvpAqBQvr0BduHDPVqssM5KIyIO_hHa4/view?usp=sharing)  
**sh demo.sh** command will execute FINIST on the LRURec model and LastFM dataset (standard training for 50 epochs + fine-tuning for 50 epochs). 
The demo execution will generate average stability and next-item metrics of LRURec + FINEST model across 3 different runs with 3 different random seeds.

Usage
---------------

The detailed execution procedure of **FINEST** is given as follows.

1) Install all required libraries by "pip install -r requirements.txt".
2) "src/[model_name]_main.py" includes data preparation/input perturbation code, while "src/[model_name]_trainer.py" contains training/evaluation code of a recommendation model.
3) "python src/[model_name].py [arguments]" will execute FINEST on the target recommendation model with arguments.

Note that the fine-tuning speed can be significantly improved by optimizing perturbation simulation part.

Demo
---------------
To run the demo, please follow the following procedure. **FINEST** demo will be executed with the LastFM dataset.

	1. Check permissions of files (if not, use the command "chmod 777 *")
	2. Execute "sh demo.sh"
	3. Check "FINEST_LRURec_LastFM" for the demo result of FINEST on LRURec and LastFM dataset

Abstract
---------------
Modern recommender systems may output considerably different recommendations due to small perturbations in the training data. Changes in the data from a single user will alter the recommendations as well as the recommendations of other users. In applications like healthcare, housing, and finance, this sensitivity can have adverse effects on user experience. We propose a method to stabilize a given recommender system against such perturbations. This is a challenging task due to (1) the lack of a “reference” rank list that can be used to anchor the outputs; and (2) the computational challenges in ensuring the stability of rank lists with respect to all possible perturbations of training data. Our method, FINEST, overcomes these challenges by obtaining reference rank lists from a given recommendation model and then fine-tuning the model under simulated perturbation scenarios with rank-preserving regularization on sampled items. Our experiments on real-world datasets demonstrate that FINEST can ensure that recommender models output stable recommendations under a wide range of different perturbations without compromising next-item prediction accuracy.
