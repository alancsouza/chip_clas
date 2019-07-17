# CHIP-clas
CHIP-clas is a machine learning classifier that uses the data structure of the Gabriel Graph to optimally separate classes
in binary classification problems. See for instance [Distance-based large margin classifier suitable for integrated circuit implementation](https://digital-library.theiet.org/content/journals/10.1049/el.2015.1644)

The NN-clas method, which is a simplification of the CHIP-clas classification stage, computing the nearest distance from 
the support edges set, is used to classify new data.
The NN-clas paper can be seen [here](http://cbic2017.org/papers/cbic-paper-33.pdf) (Portuguese)

A parallel computing technique is also implemented in order to scale de CHIP-clas algorithm to large scale problems.
See, for instance the paper: [Improving the Efficiency of Gabriel Graph-based Classifiers for Hardware-optimized Implementations] (https://ieeexplore.ieee.org/abstract/document/8730227)
