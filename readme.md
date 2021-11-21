# Estimating PageRank with inductive capability of Graph Neural Networks and zone partitioning

##### Suleyman Suleymanzade

##### Institute of information Technology

##### Azerbaijan National Academy of Science

##### Baku, Azerbaijan

##### suleyman.suleymanzade.nicat@gmail.com

### Abstract

    In this paper we proposed to use zone partitioning strategies for computing PageRank parameters in retrieval systems. The zone approach based on the idea to use multiple neural networks for classify rank data in graph-based structures. The crawled web pages are fragmented into three distinct zones. The core zone used for train GNNs; in this zone the labels are known. It covered with undiscovered zone, where classifiers labeling node parameters. The most interesting part is the intersection zone, that represent the set of nodes that belong to more than one undiscovered zone. The experiments show that probability of classifying the true labels in the intersection zones via aggregating the results of multiple classifiers in some cases is higher than in undiscovered zones.

### Setup:

create virtual environment by `virtualenv venv`
then acces to the virtualenv - Linux users `source venv/bin/activate` - Windows users `venv\Scripts\activate`
next install all important python packages by - `pip install -r requirements.txt to install all packages`
then run jupyter notebook in cmd to run the jupyter notebook - `jupyter notebook`
