# DeepSpike

This is a Python package for spike inference from calcium imaging data using deep neural networks that are trained unsupervised with variational autoencoders.

The details of the algorithm are presented in the paper [Fast amortized inference of neural activity from calcium imaging data with variational autoencoders (Speiser, Yan, Archer, Buesing, Turaga and Macke)](https://papers.nips.cc/paper/6991-fast-amortized-inference-of-neural-activity-from-calcium-imaging-data-with-variational-autoencoders)

The code requires [Theano](https://github.com/Theano/Theano) and [Lasagne](https://github.com/Lasagne/Lasagne).

The repository includes four notebooks that show how the algorithm is used:

  * EX1: Training a CNN on data simulated from a simple linear model of fluorescence dynamics.
  * EX2: Training a RNN on data simulated from a nonlinear model of fluorescence dynamics.
  * EX3: Training a RNN on publically available real data.
  * GC6s_prep: Preprocessing of the real data.
  
While the CNN can be trained on a CPU in reasonable time, training the RNN requires a GPU.
