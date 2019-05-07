# ActiveLink

## Dataset
`data/<dataset_name>`

Each dataset contains 3 files: `train.txt`, `test.txt` and `valid.txt`
Each file contains a list of triples, one triple per line.
Triple format: `<entity_1>\t<relation>\t<entity_2>`

Before the first usage the dataset should be preprocessed:

`python preprocess.py <dataset_name>`

This step generates 6 files:
* Training, validation and testing sets grouped into "multitriples" (one head entity is combined with all its tail entities in one triple) with added reverse triples. 

  Example:
    ```angular2html
      train.txt:
    /m/09sh8k	/film/film/other_crew./film/film_crew_gig/film_crew_role	/m/09vw2b7
    /m/09sh8k	/film/film/other_crew./film/film_crew_gig/film_crew_role	/m/09zzb8
    
    e1rel_to_e2_train.json:
    {"e1": "/m/09sh8k", "e2": "None", "rel": "/film/film/other_crew./film/film_crew_gig/film_crew_role", "rel_eval": "None", "e2_multi1": "/m/09zzb8 /m/09vw2b7", "e2_multi2": "None"}
    {"e1": "/m/09vw2b7", "e2": "None", "rel": "/film/film/other_crew./film/film_crew_gig/film_crew_role_reverse", "rel_eval": "None", "e2_multi1": "/m/09sh8k", "e2_multi2": "None"}
    {"e1": "/m/09zzb8", "e2": "None", "rel": "/film/film/other_crew./film/film_crew_gig/film_crew_role_reverse", "rel_eval": "None", "e2_multi1": "/m/09sh8k", "e2_multi2": "None"}
    
    ```
* Mapping from entity/relation label to id


#### Embeddings
For clustering entities (Structured Uncertainty sampling, see Section 3.2 of the paper for more details) we need to train their embeddings beforehand. We used the [TransE model](https://github.com/thunlp/KB2E/tree/master/TransE) with the following parameters:
* method: bern
* embedding size: 100
* learning rate: 0.001
* margin: 1 

Details on TransE: *Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. Learning Entity and Relation Embeddings for Knowledge Graph Completion. The 29th AAAI Conference on Artificial Intelligence (AAAI'15)* 

**NB** TransE requires mapping from entity/relation label to id. Use the `entity2id.txt` and `relation2id.txt` files generated at the preprocessing step. 

## Running a model
#### Parameters
You can configure your run via command line arguments:

    --al-epochs
        number of iterations of active learning: (dataset_size * fraction_used) / sample_size
    --batch-size
        training batch size
    --dataset
        name of dataset
    --embedding-dim
        number of embedding dimensions for entities and relations 
    --early-stop-threshold
        stop training when trigger value is above this threshold (see below)
    --eval-rate
        monitor model performance each N epochs (see below)
    --inner-lr
        learning rate for inner update in meta-incremental training
    --lr
        learning rate (meta-incremental training: learning rate for meta update)
    --lr-decay
        learning rate decay
    --model
        link prediction model, two options possible: ConvE or MLP
    --n-clusters
        number of clusters for Structured Uncertainty sampling
    --sample-size
        number of training examples per one AL iteration
    --sampling-mode
        random, uncertainty, structured or structured-uncertainty
    --training-mode
        retrain, incremental or meta-incremental
    --window-size
        size of the window for meta-incremental training
        
To reproduce the paper results for FB15k-237:
`python main.py --dataset FB15k-237 --model ConvE` (all the other parameters have right default values)

#### Early Stopping
We use early stopping at each iteration of active learning.
As a trigger we use the following formula:

`(100 * (MR / MR_opt - 1))`,

where MR is a mean rank after the current training epoch, and MR_opt is the best mean rank achieved on the previous training epochs within the same active learning iteration.

#### Evaluation Rate
Since active learning use a small fraction of a dataset at each iteration, the overall number of training epochs is much bigger for the active learning setup compared to a traditional supervised approach (in fact one iteration of active learning is comparable to the full training cycle of non-active learning in terms of the number of training epochs).
For time efficiency we do not evaluate model performance after each epoch but rather after each _<eval-rate>_ epochs.
  
#### Important Note
The library requires pytorch version 0.3.1.
For newer versions some migration updates might be needed. 

## References
For the full method description and experimental results please refer to our paper: 

Natalia Ostapuk, Jie Yang, and Philippe Cudre-Mauroux. “ActiveLink: Deep Active Learning for Link Prediction in Knowledge Graphs.” In Proceedings of the Web Conference (WWW 2019), 2019 [PDF](https://exascale.info/assets/pdf/ostapuk2019www.pdf)


## Acknowledgement
The model architecture as well as some valuable pieces of code are borrowed from this project: https://github.com/TimDettmers/ConvE 
