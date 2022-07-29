# Quantum language model with entanglement embedding for question answering

This repository contains TensorFlow implementation for QLM-EE.

A Pytorch implementation of the paper on https://iopscience.iop.org/article/10.1088/2058-9565/ac310f.

**Abstract:**
Quantum language models (QLMs) in which words are modeled as a quantum superposition of sememes have demonstrated a high level of model transparency and good post-hoc interpretability. Nevertheless, in the current literature, word sequences are basically modeled as a classical mixture of word states, which cannot fully exploit the potential of a quantum probabilistic description. A quantum-inspired neural network (NN) module is yet to be developed to explicitly capture the nonclassical correlations within the word sequences. We propose a NN model with a novel entanglement embedding (EE) module, whose function is to transform the word sequence into an entangled pure state representation. Strong quantum entanglement, which is the central concept of quantum information and an indication of parallelized correlations among the words, is observed within the word sequences. The proposed QLM with EE (QLM-EE) is proposed to implement on classical computing devices with a quantum-inspired NN structure, and numerical experiments show that QLM-EE achieves superior performance compared with the classical deep NN models and other QLMs on question answering (QA) datasets. In addition, the post-hoc interpretability of the model can be improved by quantifying the degree of entanglement among the word states.

## Prerequisite

Keras == 2.2.4

## Training and Testing

## License

MIT License

## Citation

If you find our work useful in your research, please consider citing:

```
@article{chen2021quantum,
  title={Quantum language model with entanglement embedding for question answering},
  author={Chen, Yiwei and Pan, Yu and Dong, Daoyi},
  journal={IEEE Transactions on Cybernetics},
  year={2021},
  publisher={IEEE}
}
```



