# **Federated Meta-Learning for Low-Resource Translation of Kirundi**

This repository contains the implementation of the algorithms and experiments described in the paper **"Federated Meta-Learning for Low-Resource Translation of Kirundi"**, presented at the *Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL 2025\)*.

## **Overview**

This project frames multilingual neural machine translation (NMT) as a federated meta-learning problem. It aims to improve translation quality for **Kirundi**, a low-resource Bantu language, by aggregating knowledge from related "parent" languages.

### **Key Methodology**

The approach combines Federated Learning with Meta-Learning techniques to produce a global model that can be rapidly personalized for Kirundi:

* **Federated Meta-Learning:** Aggregates gradients from multiple clients, each training on a different but related source language.  
* **PerFedAvg & Reptile:** Utilizes the PerFedAvg algorithm, combining FedAvg with Reptile meta-learning to enable few-shot adaptation.  
* **Optuna Optimization:** Uses Optuna to fine-tune gradient weighting during aggregation, optimizing the mixture of parent languages based on their lexical similarity and contribution to the target task.

## **Repository Structure**

* PersonalizedSeq2Seq.ipynb: The main notebook containing the model architecture, data loading logic, federated training loops, and evaluation metrics.  
* data/: Directory where translation datasets should be stored.
* raw/: Raw data
* graphs/: Figures
* results/: Result csvs
* studies/: Optuna experiments

## **Installation**

1. Clone this repository.  
2. Install the required dependencies:

pip install \-r requirements.txt

**Requirements:**

* Python 3.8+  
* numpy  
* torch  
* matplotlib  
* tqdm  
* sacrebleu  
* optuna  
* jupyter

*Note: This code requires a CUDA-enabled GPU for optimal performance, though it will fallback to CPU if necessary.*

## **Data Preparation**

The notebook expects parallel translation corpora in text format. Data should be placed in a data/ directory with the naming convention {lang1}-{lang2}.txt or specific filenames as defined in the readLangs function.

**Example Directory Structure:**

data/  
    eng-fra.txt  
    eng-ita.txt  
    eng-por.txt

## **Usage**

The primary entry point is the PersonalizedSeq2Seq.ipynb notebook. The core training logic is encapsulated in the run\_experiment function.

### **Running an Experiment**

1. Open the Jupyter Notebook.  
2. Navigate to the "Main Execution" section.  
3. Define the languages to use in the client cluster and run the experiment.

\# Example configuration from the notebook  
langs \= \['fra', 'ita', 'por'\]   
run\_experiment(langs, rounds=5)

This process will:

1. Initialize Encoder-Decoder models for each client language.  
2. Perform local training on parent languages.  
3. Aggregate weights using Federated Averaging.  
4. Periodically use **Optuna** to optimize the weighting of client models.  
5. Evaluate BLEU scores on the target language (Kirundi).

## **Citation**

If you use this code or dataset in your research, please cite the following paper:

**Sang, K., Rabbani, T., & Zhou, T.** (2025). Federated Meta-Learning for Low-Resource Translation of Kirundi. *Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL 2025\)*, 190-194.

@inproceedings{sang2025federated,  
  title={Federated Meta-Learning for Low-Resource Translation of Kirundi},  
  author={Sang, Kyle and Rabbani, Tahseen and Zhou, Tianyi},  
  booktitle={Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL 2025)},  
  pages={190--194},  
  year={2025}  
}
