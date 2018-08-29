# Deep Neural Networks for Coding Hospital Discharge Summaries According to ICD-9

This work was developed in the context of a MSc thesis at Instituto Superior Técnico, University of Lisbon.

The source code in this project leverages the keras.io deep learning libray for implementing a deep neural network that combines word embeddings, recurrent units, and neural attention as mechanisms for the task of automatically assigning ICD-9 diagnostic codes, by analyzing free-text descriptions within patient discharge summaries.

This neural network also explores the hierarchical nature of the input data, by building representations from the sequences of words within individual fields, which are then combined according to the sequences of fields that compose the input. This part of the neural network takes it inspiration on the model advanced by Yang et al. (2016)

    @inproceedings{yang2016hierarchical,
      title={Hierarchical Attention Networks for Document Classification},
      author={Yang, Zichao and Yang, Diyi and Dyer, Chris and He, Xiaodong and Smola, Alexander J and Hovy, Eduard H},
      booktitle={Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics},
      year={2016},
      url={https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf}
    }

Moreover, a mechanism for initializing the weights of the final nodes of the network is also used, leveraging co-occurrences between classes togheter with the hierarchical structure of ICD-9.

The code was tested with Pyhton 3.6.0 and Keras 2.1.5

### Training a model

1. Using a `.txt` file with your dataset (see `dataset_example_hba_full.txt`), execute the `clinical_coding_dnn.py` indicating the dataset file directory in `line 84` of the code.

2. After the training is complete, the model saves five `.txt` files with the outputs for each hierarchical level (e.g., `pred_full_nmf.txt`)

3. The following files are saved: `modelo_full_nmf.h5`, `DICT.npy`, `MAIN.npy`, `FULL_CODES.npy` and `BLOCKS.npy`. These are the files needed to load the model.
