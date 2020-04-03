---
layout: article
title: Summary of graph data mining algorithms (deep model)
date: 2020-03-15 00:10:00 +0800
tags: [Summary, Graph, Link prediction, Node Classification, Graph Classification, Adversarial]
categories: blog
pageview: true
key: Summary-of-graph-data-mining-algorithms
---



# Graph Algorithm

A collection of graph task models, covering node classification, link prediction, graph classification and multi-task models **with reference implementations**.



## Reference

- https://paperswithcode.com/area/graphs
- https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph



## Contents

1. [Node Classification](#Node Classification)
2. [Link Prediction](#Link Prediction)
3. [Multi-task Model](#Multi-task Model)
4. [Graph Classification](#Graph Classification)
5. [Adversarial Learning](#Adversarial Learning)



## Node Classification

- **[GCN]** **Semi-Supervised Classification with Graph Convolutional Networks**
  - [[Paper]](https://arxiv.org/abs/1609.02907)
  - [[Tensorflow]](https://github.com/tkipf/gcn)
  - [[Keras]](https://github.com/tkipf/keras-gcn)
- **[GraphSAGE] Inductive representation learning on large graphs**
  - [[Paper]](https://arxiv.org/pdf/1706.02216.pdf)
  - [[Python Reference]](https://github.com/chauhanjatin10/GraphsFewShot)
  - [[Pytorch 1]](https://github.com/williamleif/graphsage-simple)
  - [[Pytorch 2]](https://github.com/bkj/pytorch-graphsage) 
- **[GAT] Graph Attention Networks**
  - [[Paper]](https://arxiv.org/pdf/1710.10903v3.pdf) 
  - [[Tensorflow]](https://github.com/PetarV-/GAT)
  - [[Keras]](https://github.com/danielegrattarola/keras-gat)
  - [[Pytroch]](https://github.com/Diego999/pyGAT)
- **[LGCN] Large-Scale Learnable Graph Convolutional Networks**
  - [[Paper]](https://arxiv.org/abs/1808.03965)
  - [[Tensorflow]](https://github.com/divelab/lgcn/) 





## Link prediction

- **[GAE/VGAE] Variational Graph Auto-Encoders**
  - [[Paper]](https://arxiv.org/abs/1611.07308)
  - [[Tensorflow]](https://github.com/tkipf/gae)
- **[DeepLinker] Link Prediction via Graph Attention Network**
  - [[Paper]](https://arxiv.org/abs/1910.04807)
  - [[Pytorch]](https://github.com/Villafly/DeepLinker) 
- **[MTGAE] Multi-Task Graph Autoencoders**
  - TASK: link prediction, Node classification
  - [[Paper]](https://arxiv.org/pdf/1811.02798.pdf) 
  - [[Keras]](https://github.com/vuptran/graph-representation-learning) 
- **[ARGA] Adversarially Regularized Graph Autoencoder for Graph Embedding**
  - TASK: link prediction, graph clustering
  - [[Paper]](https://arxiv.org/pdf/1802.04407v2.pdf) 
  - [[Tensorflow]](https://github.com/Ruiqi-Hu/ARGA)
- **[GraphStar] Graph Star Net for Generalized Multi-Task Learning**
  - TASK: link prediction, Node classification, text classification, graph classification, sentiment analysis
  - [[Paper]](https://paperswithcode.com/paper/graph-star-net-for-generalized-multi-task-1)
  - [[Pytorch]](https://github.com/graph-star-team/graph_star)





## Multi-task Model

- **[MTGAE] Multi-Task Graph Autoencoders**
  - TASK: link prediction, Node classification
  - [[Paper]](https://arxiv.org/pdf/1811.02798.pdf) 
  - [[Keras]](https://github.com/vuptran/graph-representation-learning) 
- **[ARGA] Adversarially Regularized Graph Autoencoder for Graph Embedding**
  - TASK: link prediction, graph clustering
  - [[Paper]](https://arxiv.org/pdf/1802.04407v2.pdf) 
  - [[Tensorflow]](https://github.com/Ruiqi-Hu/ARGA)
- **[GraphStar] Graph Star Net for Generalized Multi-Task Learning**
  - TASK: link prediction, Node classification, text classification, graph classification, sentiment analysis
  - [[Paper]](https://paperswithcode.com/paper/graph-star-net-for-generalized-multi-task-1)
  - [[Pytorch]](https://github.com/graph-star-team/graph_star)





## Graph Classification

- **Few-shot Learning on Graphs Via Super-Classes Based on Graph Spectral Measures (ICLR 2020)**
  - Jatin Chauhan, Deepak Nathani, Manohar Kaul
  - [[Paper]](https://openreview.net/forum?id=Bkeeca4Kvr)
  - [[Python Reference]](https://github.com/chauhanjatin10/GraphsFewShot)
  
- **Memory-Based Graph Networks (ICLR 2020)**
  - Amir Hosein Khasahmadi, Kaveh Hassani, Parsa Moradi, Leo Lee, Quaid Morris
  - [[Paper]](https://openreview.net/forum?id=r1laNeBYPB)
  - [[Python Reference]](https://github.com/amirkhas/GraphMemoryNet)

- **A Fair Comparison of Graph Neural Networks for Graph Classification (ICLR 2020)**
  - Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli
  - [[Paper]](https://openreview.net/pdf?id=HygDF6NFPB)
  - [[Python Reference]](https://github.com/diningphil/gnn-comparison)

- **StructPool: Structured Graph Pooling via Conditional Random Fields (ICLR 2020)**
  - Hao Yuan, Shuiwang Ji
  - [[Paper]](https://openreview.net/forum?id=BJxg_hVtwH)
  - [[Python Reference]](https://github.com/Nate1874/StructPool)
  
- **InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization (ICLR 2020)**
  - Fan-yun Sun, Jordan Hoffman, Vikas Verma, Jian Tang
  - [[Paper]](https://openreview.net/pdf?id=r1lfF2NYvH)
  - [[Python Reference]](https://github.com/fanyun-sun/InfoGraph)
  
- **Convolutional Kernel Networks for Graph-Structured Data (ArXiV 2020)**
  - Dexiong Chen, Laurent Jacob, Julien Mairal
  - [[Paper]](https://arxiv.org/abs/2003.05189)
  - [[Python Reference]](https://github.com/claying/GCKN)

- **Building Attention and Edge Convolution Neural Networks for Bioactivity and Physical-Chemical Property Prediction (AAAI 2020)**
  - Michael Withnall, Edvard Lindelöf, Ola Engkvist, Hongming Chen
  - [[Paper]](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0407-y)
  - [[Python Reference]](https://github.com/edvardlindelof/graph-neural-networks-for-drug-discovery)
  
- **GSSNN: Graph Smoothing Splines Neural Network (AAAI 2020)**
  - Shichao Zhu, Lewei Zhou, Shirui Pan, Chuan Zhou, Guiying Yan, Bin Wang 
  - [[Paper]](https://shiruipan.github.io/publication/aaai-2020-zhu)
  - [[Python Reference]](https://github.com/CheriseZhu/GSSNN)
  
- **Discriminative Structural Graph Classification (ArXiV 2019)**
  - Younjoo Seo, Andreas Loukas, Nathanaël Perraudin
  - [[Paper]](https://arxiv.org/abs/1905.13422)
  - [[Python Reference]](https://github.com/youngjoo-epfl/DSGC)
  
- **Graph Classification with Automatic Topologically-Oriented Learning (ArXiV 2019)**
  - Martin Royer, Frédéric Chazal, Clément Levrard, Yuichi Ike, Yuhei Umeda
  - [[Paper]](https://arxiv.org/pdf/1909.13472.pdf)
  - [[Python Reference]](https://github.com/martinroyer/atol)
  - [[Python]](https://github.com/giotto-ai/graph_classification_with_atol)
  
- **Graph Convolutional Networks with EigenPooling (KDD 2019)**
  - Yao Ma, Suhang Wang, Charu C Aggarwal, Jiliang Tang
  - [[Paper]](https://arxiv.org/pdf/1904.13107.pdf)
  - [[Python Reference]](https://github.com/alge24/eigenpooling)

- **Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels (NeurIPS 2019)**
  - Simon S. Du, Kangcheng Hou, Barnabás Póczos, Ruslan Salakhutdinov, Ruosong Wang, Keyulu Xu
  - [[Paper]](https://arxiv.org/abs/1905.13192)
  - [[Python Reference]](https://github.com/KangchengHou/gntk)

- **Molecule Property Prediction Based on Spatial Graph Embedding (Journal of Cheminformatics Models 2019)**
  - Xiaofeng Wang, Zhen Li, Mingjian Jiang, Shuang Wang, Shugang Zhang, Zhiqiang Wei
  - [[Paper]](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00410)
  - [[Python Reference]](https://github.com/1128bian/C-SGEN)
  
- **Unsupervised Universal Self-Attention Network for Graph Classification (Arxiv 2019)**
  - Dai Quoc Nguyen, Tu Dinh Nguyen, and Dinh Phun
  - [[Paper]](https://arxiv.org/abs/1909.11855)
  - [[Python Reference]](https://github.com/daiquocnguyen/U2GNN)

- **Fast Training of Sparse Graph Neural Networks on Dense Hardware (Arxiv 2019)**
  - Matej Balog, Bart van Merriënboer, Subhodeep Moitra, Yujia Li, Daniel Tarlow
  - [[Paper]](https://arxiv.org/abs/1906.11786)
  - [[Python Reference]](https://github.com/anonymous-authors-iclr2020/fast_training_of_sparse_graph_neural_networks_on_dense_hardware)
  
- **Hierarchical Representation Learning in Graph Neural Networks with Node Decimation Pooling (Arxiv 2019)**
  - Filippo Maria Bianchi, Daniele Grattarola, Lorenzo Livi, Cesare Alippi
  - [[Paper]](https://arxiv.org/abs/1910.11436)
  - [[Python Reference]](https://github.com/danielegrattarola/decimation-pooling)
  
- **Are Powerful Graph Neural Nets Necessary? A Dissection on Graph Classification (Arxiv 2019)**
  - Ting Chen, Song Bian, Yizhou Sun
  - [[Paper]](https://arxiv.org/abs/1905.04579)
  - [[Python Reference]](https://github.com/Waterpine/vis_network) 

- **Learning Aligned-Spatial Graph Convolutional Networks for Graph Classification (ECML-PKDD 2019)**
  - Lu Bai, Yuhang Jiao, Lixin Cui, Edwin R. Hancock
  - [[Paper]](https://arxiv.org/abs/1904.04238)
  - [[Python Reference]](https://github.com/baiuoy/ASGCN_ECML-PKDD2019) 

- **Relational Pooling for Graph Representations (ICML 2019)**
  - Ryan L. Murphy, Balasubramaniam Srinivasan, Vinayak Rao, Bruno Ribeiro
  - [[Paper]](https://arxiv.org/abs/1903.02541)
  - [[Python Reference]](https://github.com/PurdueMINDS/RelationalPooling)

- **Ego-CNN: Distributed, Egocentric Representations of Graphs for Detecting Critical Structure (ICML 2019)**
  - Ruo-Chun Tzeng, Shan-Hung Wu
  - [[Paper]](http://proceedings.mlr.press/v97/tzeng19a/tzeng19a.pdf)
  - [[Python Reference]](https://github.com/rutzeng/EgoCNN)

- **Self-Attention Graph Pooling (ICML 2019)**
  - Junhyun Lee, Inyeop Lee, Jaewoo Kang
  - [[Paper]](https://arxiv.org/abs/1904.08082)
  - [[Python Reference]](https://github.com/inyeoplee77/SAGPool)

- **Variational Recurrent Neural Networks for Graph Classification (ICLR 2019)**
  - Edouard Pineau, Nathan de Lara
  - [[Paper]](https://arxiv.org/abs/1902.02721)
  - [[Python Reference]](https://github.com/edouardpineau/Variational-Recurrent-Neural-Networks-for-Graph-Classification)

- **Crystal Graph Neural Networks for Data Mining in Materials Science (Arxiv 2019)**
  - Takenori Yamamoto
  - [[Paper]](https://storage.googleapis.com/rimcs_cgnn/cgnn_matsci_May_27_2019.pdf)
  - [[Python Reference]](https://github.com/Tony-Y/cgnn)

- **Explainability Techniques for Graph Convolutional Networks (ICML 2019 Workshop)**
  - Federico Baldassarre, Hossein Azizpour
  - [[Paper]](https://128.84.21.199/pdf/1905.13686.pdf)
  - [[Python Reference]](https://github.com/gn-exp/gn-exp)

- **Semi-Supervised Graph Classification: A Hierarchical Graph Perspective (WWW 2019)**
  - Jia Li, Yu Rong, Hong Cheng, Helen Meng, Wenbing Huang, and Junzhou Huang
  - [[Paper]](https://arxiv.org/pdf/1904.05003.pdf)
  - [[Python Reference]](https://github.com/benedekrozemberczki/SEAL-CI)

- **Capsule Graph Neural Network (ICLR 2019)**
  - Zhang Xinyi and Lihui Chen
  - [[Paper]](https://openreview.net/forum?id=Byl8BnRcYm)
  - [[Python Reference]](https://github.com/benedekrozemberczki/CapsGNN)

- **How Powerful are Graph Neural Networks? (ICLR 2019)**
  - Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
  - [[Paper]](https://arxiv.org/abs/1810.00826)
  - [[Python Reference]](https://github.com/weihua916/powerful-gnns)

- **Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks (AAAI 2019)**
  - Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe
  - [[Paper]](https://arxiv.org/pdf/1810.02244v2.pdf)
  - [[Python Reference]](https://github.com/k-gnn/k-gnn)

- **Capsule Neural Networks for Graph Classification using Explicit Tensorial Graph Representations (Arxiv 2019)**
  - Marcelo Daniel Gutierrez Mallea, Peter Meltzer, and Peter J Bentley
  - [[Paper]](https://arxiv.org/pdf/1902.08399v1.pdf)
  - [[Python Reference]](https://github.com/BraintreeLtd/PatchyCapsules)
  
- **Mapping Images to Scene Graphs with Permutation-Invariant Structured Prediction (NIPS 2019)**
  - Roei Herzig, Moshiko Raboh, Gal Chechik, Jonathan Berant, Amir Globerson
  - [[Paper]](https://arxiv.org/abs/1802.05451)
  - [[Python Reference]](https://github.com/shikorab/SceneGraph)
  
- **Fast and Accurate Molecular Property Prediction: Learning Atomic Interactions and Potentials with Neural Networks (The Journal of Physical Chemistry Letters 2018)**
  - Masashi Tsubaki and Teruyasu Mizoguchi
  - [[Paper]](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b01837)
  - [[Python Reference]](https://github.com/masashitsubaki/molecularGNN_3Dstructure)
  
- **Machine Learning for Organic Cage Property Prediction (Chemical Matters 2018)**
  - Lukas Turcani, Rebecca Greenway, Kim Jelfs
  - [[Paper]](https://pubs.acs.org/doi/10.1021/acs.chemmater.8b03572)
  - [[Python Reference]](https://github.com/qyuan7/Graph_Convolutional_Network_for_cages)
  
- **Three-Dimensionally Embedded Graph Convolutional Network for Molecule Interpretation (Arxiv 2018)**
  - Hyeoncheol Cho and Insung. S. Choi
  - [[Paper]](https://arxiv.org/abs/1811.09794)
  - [[Python Reference]](https://github.com/blackmints/3DGCN)

- **Learning Graph-Level Representations with Recurrent Neural Networks (Arxiv 2018)**
  - Yu Jin and Joseph F. JaJa
  - [[Paper]](https://arxiv.org/pdf/1805.07683v4.pdf)
  - [[Python Reference]](https://github.com/yuj-umd/graphRNN)

- **Graph Capsule Convolutional Neural Networks (ICML 2018)**
  - Saurabh Verma and Zhi-Li Zhang
  - [[Paper]](https://arxiv.org/abs/1805.08090)
  - [[Python Reference]](https://github.com/vermaMachineLearning/Graph-Capsule-CNN-Networks)

- **Graph Classification Using Structural Attention (KDD 2018)**
  - John Boaz Lee, Ryan Rossi, and Xiangnan Kong
  - [[Paper]](http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf)
  - [[Python Pytorch Reference]](https://github.com/benedekrozemberczki/GAM)

- **Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation (NIPS 2018)**
  - Jiaxuan You, Bowen Liu, Rex Ying, Vijay Pande, and Jure Leskovec
  - [[Paper]](https://arxiv.org/abs/1806.02473)
  - [[Python Reference]](https://github.com/bowenliu16/rl_graph_generation)

- **Hierarchical Graph Representation Learning with Differentiable Pooling (NIPS 2018)**
  - Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton and Jure Leskovec
  - [[Paper]](http://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf)
  - [[Python Reference]](https://github.com/rusty1s/pytorch_geometric)

- **Contextual Graph Markov Model: A Deep and Generative Approach to Graph Processing (ICML 2018)**
  - Davide Bacciu, Federico Errica, and Alessio Micheli
  - [[Paper]](https://arxiv.org/pdf/1805.10636.pdf)
  - [[Python Reference]](https://github.com/diningphil/CGMM)

- **MolGAN: An Implicit Generative Model for Small Molecular Graphs (ICML 2018)**
  - Nicola De Cao and Thomas Kipf
  - [[Paper]](https://arxiv.org/pdf/1805.11973.pdf)
  - [[Python Reference]](https://github.com/nicola-decao/MolGAN)

- **Deeply Learning Molecular Structure-Property Relationships Using Graph Attention Neural Network (2018)**
  - Seongok Ryu, Jaechang Lim, and Woo Youn Kim    
  - [[Paper]](https://arxiv.org/abs/1805.10988)
  - [[Python Reference]](https://github.com/SeongokRyu/Molecular-GAT)

- **Compound-Protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics 2018)**
  - Masashi Tsubaki, Kentaro Tomii, and Jun Sese
  - [[Paper]](https://academic.oup.com/bioinformatics/article/35/2/309/5050020)
  - [[Python Reference]](https://github.com/masashitsubaki/CPI_prediction)
  - [[Python Reference]](https://github.com/masashitsubaki/GNN_molecules)
  - [[Python Alternative ]](https://github.com/xnuohz/GCNDTI)

- **Learning Graph Distances with Message Passing Neural Networks (ICPR 2018)**
  - Pau Riba, Andreas Fischer, Josep Llados, and Alicia Fornes
  - [[Paper]](https://ieeexplore.ieee.org/abstract/document/8545310)
  - [[Python Reference]](https://github.com/priba/siamese_ged)

- **Edge Attention-based Multi-Relational Graph Convolutional Networks (2018)**
  - Chao Shang, Qinqing Liu, Ko-Shin Chen, Jiangwen Sun, Jin Lu, Jinfeng Yi and Jinbo Bi  
  - [[Paper]](https://arxiv.org/abs/1802.04944v1)
  - [[Python Reference]](https://github.com/Luckick/EAGCN)

- **Commonsense Knowledge Aware Conversation Generation with Graph Attention (IJCAI-ECAI 2018)**
  - Hao Zhou, Tom Yang, Minlie Huang, Haizhou Zhao, Jingfang Xu and Xiaoyan Zhu
  - [[Paper]](http://coai.cs.tsinghua.edu.cn/hml/media/files/2018_commonsense_ZhouHao_3_TYVQ7Iq.pdf)
  - [[Python Reference]](https://github.com/tuxchow/ccm)

- **Residual Gated Graph ConvNets (ICLR 2018)**
  - Xavier Bresson and Thomas Laurent
  - [[Paper]](https://arxiv.org/pdf/1711.07553v2.pdf)
  - [[Python Pytorch Reference]](https://github.com/xbresson/spatial_graph_convnets)

- **An End-to-End Deep Learning Architecture for Graph Classification (AAAI 2018)**
  - Muhan Zhang, Zhicheng Cui, Marion Neumann and Yixin Chen
  - [[Paper]](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf)
  - [[Python Tensorflow Reference]](https://github.com/muhanzhang/DGCNN)    
  - [[Python Pytorch Reference]](https://github.com/muhanzhang/pytorch_DGCNN)
  - [[MATLAB Reference]](https://github.com/muhanzhang/DGCNN)
  - [[Python Alternative]](https://github.com/leftthomas/DGCNN)
  - [[Python Alternative]](https://github.com/hitlic/DGCNN-tensorflow)

- **SGR: Self-Supervised Spectral Graph Representation Learning (KDD DLDay 2018)**
  - Anton Tsitsulin, Davide Mottin, Panagiotis Karra, Alex Bronstein and Emmanueal Müller
  - [[Paper]](https://arxiv.org/abs/1807.02839)
  - [[Python Reference]](http://mott.in/publications/others/sgr/)

- **Deep Learning with Topological Signatures (NIPS 2017)**
  - Christoph Hofer, Roland Kwitt, Marc Niethammer, and Andreas Uhl
  - [[paper]](https://arxiv.org/abs/1707.04041)
  - [[Python Reference]](https://github.com/c-hofer/nips2017)

- **Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs (CVPR 2017)**
  - Martin Simonovsky and Nikos Komodakis
  - [[paper]](https://arxiv.org/pdf/1704.02901v3.pdf)
  - [[Python Reference]](https://github.com/mys007/ecc)

- **Deriving Neural Architectures from Sequence and Graph Kernels (ICML 2017)**
  - Tao Lei, Wengong Jin, Regina Barzilay, and Tommi Jaakkola
  - [[Paper]](https://arxiv.org/abs/1705.09037)
  - [[Python Reference]](https://github.com/taolei87/icml17_knn)

- **Protein Interface Prediction using Graph Convolutional Networks (NIPS 2017)**
  - Alex Fout, Jonathon Byrd, Basir Shariat and Asa Ben-Hur
  - [[Paper]](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks)
  - [[Python Reference]](https://github.com/fouticus/pipgcn)

- **Graph Classification with 2D Convolutional Neural Networks (2017)**
  - Antoine J.-P. Tixier, Giannis Nikolentzos, Polykarpos Meladianos and Michalis Vazirgiannis
  - [[Paper]](https://arxiv.org/abs/1708.02218)
  - [[Python Reference]](https://github.com/Tixierae/graph_2D_CNN)

- **CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters (IEEE TSP 2017)**
  - Ron Levie, Federico Monti, Xavier Bresson, Michael M. Bronstein
  - [[Paper]](https://arxiv.org/pdf/1705.07664v2.pdf)
  - [[Python Reference]](https://github.com/fmonti/CayleyNet)

- **Semi-Supervised Learning of Hierarchical Representations of Molecules Using Neural Message Passing (2017)**
  - Hai Nguyen, Shin-ichi Maeda, Kenta Oono
  - [[Paper]](https://arxiv.org/pdf/1711.10168.pdf)
  - [[Python Reference]](https://github.com/pfnet-research/hierarchical-molecular-learning)

- **Kernel Graph Convolutional Neural Networks (2017)**
  - Giannis Nikolentzos, Polykarpos Meladianos, Antoine Jean-Pierre Tixier, Konstantinos Skianis, Michalis Vazirgiannis
  - [[Paper]](https://arxiv.org/pdf/1710.10689.pdf)
  - [[Python Reference]](https://github.com/giannisnik/cnn-graph-classification)

- **Deep Topology Classification: A New Approach For Massive Graph Classification (IEEE Big Data 2016)**
  - Stephen Bonner, John Brennan, Georgios Theodoropoulos, Ibad Kureshi, Andrew Stephen McGough
  - [[Paper]](https://ieeexplore.ieee.org/document/7840988/)
  - [[Python Reference]](https://github.com/sbonner0/DeepTopologyClassification)

- **Learning Convolutional Neural Networks for Graphs (ICML 2016)**
  - Mathias Niepert, Mohamed Ahmed, Konstantin Kutzkov
  - [[Paper]](https://arxiv.org/abs/1605.05273)
  - [[Python Reference]](https://github.com/tvayer/PSCN)

- **Gated Graph Sequence Neural Networks (ICLR 2016)**
  - Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel
  - [[Paper]](https://arxiv.org/abs/1511.05493)
  - [[Python TensorFlow]](https://github.com/bdqnghi/ggnn.tensorflow)
  - [[Python PyTorch]](https://github.com/JamesChuanggg/ggnn.pytorch)
  - [[Python Reference]](https://github.com/YunjaeChoi/ggnnmols)

- **Convolutional Networks on Graphs for Learning Molecular Fingerprints (NIPS 2015)**
  - David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P. Adams
  - [[Paper]](https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf)
  - [[Python Reference]](https://github.com/fllinares/neural_fingerprints_tf)
  - [[Python Reference]](https://github.com/jacklin18/neural-fingerprint-in-GNN)
  - [[Python Reference]](https://github.com/HIPS/neural-fingerprint)
  - [[Python Reference]](https://github.com/debbiemarkslab/neural-fingerprint-theano)



## Adversarial Learning

### Survey Paper

- **Adversarial Attacks and Defenses on Graphs: A Review and Empirical Study** 
  - [[Paper]](https://arxiv.org/abs/2003.00653) 
  - [[Pytorch]](https://github.com/DSE-MSU/DeepRobust/)



### Attack

- [[DeepRobust]](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph)

    | Attack Methods | Type            | Perturbation       | Evasion/ Poisoning | Apply Domain        | Links                                                        |
    | -------------- | --------------- | ------------------ | ------------------ | ------------------- | ------------------------------------------------------------ |
    | Nettack        | Targeted Attack | Structure Features | Both               | Node Classification | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf) |
    | FGA            | Targeted Attack | Structure          | Both               | Node Classification | [Fast Gradient Attack on Network Embedding](https://arxiv.org/pdf/1809.02797.pdf) |
    | Mettack        | Global Attack   | Structure Features | Poisoning          | Node Classification | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://openreview.net/pdf?id=Bylnx209YX) |
    | RL-S2V         | Targeted Attack | Structure          | Evasion            | Node Classification | [Adversarial Attack on Graph Structured Data](https://arxiv.org/pdf/1806.02371.pdf) |
    | PGD, Min-max   | Global Attack   | Structure          | Both               | Node Classification | [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/pdf/1906.04214.pdf) |
    | DICE           | Global Attack   | Structure          | Both               | Node Classification | [Hiding individuals and communities in a social network](https://arxiv.org/abs/1608.00375) |
    | IG-Attack      | Targeted Attack | Structure Features | Both               | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf) |
    | NIPA           | Global Attack   | Structure          | Poisoning          | Node Classification | [Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach](https://arxiv.org/pdf/1909.06543.pdf) |



- **A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models.** (AAAI 2020)
  - [[Paper]](https://arxiv.org/pdf/1908.01297.pdf)
  - [[Code]](https://github.com/SwiftieH/GFAttack)
- **Adversarial Attacks on Node Embeddings via Graph Poisoning.** (ICML 2019)
  - [[Paper]](https://arxiv.org/pdf/1809.01093.pdf)
  - [[Code]](https://github.com/abojchevski/node_embedding_attack)
- **Adversarial Attack on Graph Structured Data.** (ICML 2018). 
  - [[Paper\]](https://arxiv.org/pdf/1806.02371.pdf) 
  - [[Code\]](https://github.com/Hanjun-Dai/graph_adversarial_attack)
- **Fast Gradient Attack on Network Embedding.** (arxiv 2018). 
  - [[Paper\]](https://arxiv.org/pdf/1809.02797.pdf) 
  - [[Code\]](https://github.com/DSE-MSU/DeepRobust)
- **Adversarial Attacks on Neural Networks for Graph Data.** (KDD 2018). 
  - [[Paper\]](https://arxiv.org/pdf/1805.07984.pdf) 
  - [[Code\]](https://github.com/danielzuegner/nettack)
- **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense.** (IJCAI 2019). 
  - [[Paper\]](https://arxiv.org/pdf/1903.01610.pdf) 
  - [[Code\]](https://github.com/DSE-MSU/DeepRobust)
- **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective.** (ICJAI 2019). 
  - [[Paper\]](https://arxiv.org/pdf/1906.04214.pdf) 
  - [[Code\]](https://github.com/KaidiXu/GCN_ADV_Train)
- **Adversarial Attacks on Node Embeddings via Graph Poisoning.** (ICML 2019). 
  - [[paper\]](https://arxiv.org/pdf/1809.01093.pdf) 
  - [[code\]](https://github.com/abojchevski/node_embedding_attack)
- **Adversarial Attacks on Graph Neural Networks via Meta Learning.** (ICLR 2019). 
  - [[Paper\]](https://openreview.net/pdf?id=Bylnx209YX) 
  - [[Code\]](https://github.com/danielzuegner/gnn-meta-attack)



### Defense

- [[DeepRobust]](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph)

    | Defense Methods | Defense Type  | Apply Domain        | Links                                                        |
    | --------------- | ------------- | ------------------- | ------------------------------------------------------------ |
    | RGCN            | Gaussian      | Node Classification | [Robust Graph Convolutional Networks Against Adversarial Attacks](http://pengcui.thumedialab.com/papers/RGCN.pdf) |
    | GCN-Jaccard     | Preprocessing | Node Classification | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/pdf/1903.01610.pdf) |
    | GCN-SVD         | Preprocessing | Node Classification | [All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789?download=true) |

- **All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs.** (WSDM 2020). 

    - [[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789?download=true) 
    - [[Code\]](https://github.com/DSE-MSU/DeepRobust/)

- **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense.** (IJCAI 2019). 

    - [[Paper\]](https://arxiv.org/pdf/1903.01610.pdf) 
    - [[Code\]](https://github.com/DSE-MSU/DeepRobust)

- **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective.** *Kaidi Xu, Hongge Chen, Sijia Liu, Pin-Yu Chen, Tsui-Wei Weng, Mingyi Hong, Xue Lin.* ICJAI 2019. [[paper\]](https://arxiv.org/pdf/1906.04214.pdf) [[code\]](https://github.com/KaidiXu/GCN_ADV_Train)



### Certified Robustness 

- **Certifiable Robustness to Graph Perturbations.**  (NeurIPS 2019). 
  - [[Paper\]](https://arxiv.org/pdf/1910.14356.pdf)
  - [[Code\]](https://github.com/abojchevski/graph_cert)

- **Certifiable Robustness and Robust Training for Graph Convolutional Networks.** (KDD 2019). 
  - [[Paper\]](https://arxiv.org/pdf/1906.12269.pdf) 
  - [[Code\]](https://github.com/danielzuegner/robust-gcn)

