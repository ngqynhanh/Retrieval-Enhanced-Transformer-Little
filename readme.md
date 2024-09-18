# Retrieval-Enhanced-Transformer-Little
The repository associated with the ECAI 2024 publication titled "Retro-Li: Small-Scale Retrieval Augmented Generation Supporting Noisy Similarity Searches and  Domain Shift Generalization"

Authors: Geethan Karunaratne <kar@zurich.ibm.com>, Gentiana Rashiti

#### Installing Dependencies
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels.
Further, we have used conda as a python package manager and exported the environment specifications to `spec-file.txt`. 
You can recreate our environment by running 

```
$ conda create --name labml-retro --file spec-file.txt
$ conda install torchinfo
$ pip install debugpy
$ pip install labml
$ pip install labml_helpers
$ pip install labml_nn
```


## To create the datasets
We first create the FAISS index, then the jsons containing the training/validation samples with their nearest neighbors as found by the index.
### database_train.py / database_val.py
This script loads the dataset, cleans it up, tokenizes it and then creates an index based on the chunks created of the tokenized text. Due to memory constraints, the cleanup, tokenization and chunk creation must be done before we create the index. Thus the output is a retrieval database in form of a FAISS index.
If the retrieval database is the same data as the training data, we have to either
1. take disjoint subsets for training and retrieval. This also means that we have two index files, one for training (disjoint from training samples) and one for validation (contains all of the training tokens). In this case, skip_range=0 and you need to call both database_train.py and database_val.py.
In database_train.py you will also need to specify what percentage of the training data you would like to use to create the training retrieval database, then take the complement of that for dataset.py. So if you use the first 49% of wikitext to create the training retrieval database you would load
    text_train = load_dataset('wikitext', huggingface_dataset, split="train[49%:]")
in database_train.py and
    text_train = load_dataset('wikitext', huggingface_dataset, split="train[-51%:]")
in dataset.py
or
2. skip tokens and remember offsets to make sure that the neighbors aren't perfectly aligned. In this case, skip_range != 0 and you only need to call database_val.py
If the retrieval database is not the same data as the training data, you only need to call database_val.py which creates a retrieval database with all of the training data of your chosen huggingface dataset.
So once more: database_train.py creates the retrieval database to use during training **only** if there are different retrieval databases for training and validation. Else database_val.py creates the retrieval database for training and for validation.
**Important** For the inference experiments: DO NOT forget to set the centroids, code size and n probe accordingly! For these scripts use the
> retro-li/retro/configs/retro_small_model_wikitext103-gpt2-10.json
config file, which specifies 10 neighbors.
#### Huggingface
We load huggingface datasets, so if you would like to only use one specific subset of such a dataset, you need to download their python file and adjust it accordingly. For instance, we only use atticus contracts and founding documents from pile-of-law. To avoid downloading all of it, we download their "pile-of-law.py" file, and remove all datasets except these two. Then instead of calling
    text = load_dataset("pile-of-law", "atticus_contracts")
we call
    text = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "atticus_contracts")
#### Example
    python database_train.py \
    --index_filepath 'test/retro_bbc_l2_gpt2_train.index' \
    --config_filepath 'configs/retro_small_model_wikitext103-gpt2-10.json' \
    --huggingface_dataset 'SetFit/bbc-news' \
    --intermediate_results_dir 'test' \
    --truncation "True"
Where the index filepath is where you would like to save the index to, the config filepath is which configuration file you want to use and the huggingface dataset describes what dataset you would like to load. For "database_train.py" you must be very detailed here, whereas for "database_val.py" you can be more general (for instance instead of writing `SetFit/bbc-news` you can write `bbc`). Since we use very specific versions for our inference experiments, each version is "hard-coded" into the database_val.py script.
The intermediate results directory is a time saving measure. Since we re-compute the tokenized text later on, we can instead save it here and then load it for the dataset creation. If you don't want to use it, just keep it an empty string.
Truncation refers to if we want our samples during tokenization to be truncated to 1024 tokens or not. This is a nice and simple way to keep the samples small. For most datasets the samples are already less than 1024 tokens long.
### dataset.py
This script creates the training data. This takes some dataset X, tokenizes and chunks it, and finds the nearest neighbors in the index Y created by database_train.py / database_val.py, then creates a json Z with the training data in the form of [source, target, neighbors] or optionally [source, target, neighbors, distance_matrix].
**Important** : If you only want to create the validation dataset for inference for instance, set flattened_text_tokenized_path_train to an empty string when you run the script. For these scripts use the
> retro-li/configs/retro_small_model_wikitext103-gpt2-10.json
config file, which specifies 10 neighbors.
#### Example
This is an example script to run dataset.py

    python dataset.py \
    --train_index_filepath 'test/retro_wiki103_l2_gpt2_-trunc_sentemb.index' \
    --val_index_filepath 'test/retro_wiki103_l2_gpt2_-trunc_sentemb.index' \
    --train_dataset_filepath 'test/retro_train_dataset_wikitext103_l2_gpt2_10neighbors_2.json' \
    --val_dataset_filepath 'test/retro_val_dataset_wikitext103_l2_gpt2_10neighbors_2.json' \
    --config_filepath 'configs/retro_small_model_wikitext103-gpt2-10.json' \
    --flattened_text_tokenized_path_train "None" \
    --flattened_text_tokenized_path_val "None" \
    --huggingface_dataset 'wikitext-103-raw-v1'
The index files refer to the FAISS indexes, so the retrieval databases. Depending on the setup, we either have different databases for training and validation, or the same one for both.
The dataset filepaths refer to the jsons we would like to create using this code.
The config filepath describes which configuration we use for our datasets.
The "flattened_text_tokenized_path" for train and val refers to a time / memory saving measure. 
Namely, since tokenizing and flattening the datasets is very time-consuming, and we already need to do it to create the index, we can save that intermediate result when we call database_train.py and/or database_val.py and load it here. If nothing is specified, we simply re-compute it.
The huggingface dataset refers to the dataset used for training.
## To train
The train.py script encompasses all training modes (on, off, on-reg, retrofit, minifit).
Training can be started via a combination of a script in the scripts directory and the config file you specify in that script. The scripts are for pure convenience, everything is specified in the config file (and in which training script you chose), except for what data to train and validate on, this is where you must input the json files created by dataset.py.
Once the datasets needed for training / validation are generated all you need to do is run

    python train.py \
    --random_seed "42" \
    --train_dataset_filepath 'test/retro_train_dataset_wikitext103_l2_gpt2_10neighbors_2.json' \
    --val_dataset_filepath 'test/retro_val_dataset_wikitext103_l2_gpt2_10neighbors_2.json' \
    --config_filepath 'configs/retro_small_model_wikitext103-gpt2-10.json' 

from retro-li directory.
If you would like to run retro-off, simply use the retro-off config file

> configs/retro_small_model_wikitext103-gpt2_retro_off.json

If you would like to run retro-on, simply run the retro-on config file

> configs/retro_small_model_wikitext103-gpt2-3.json

If you would like to run retro-on-reg, simply run the retro-on-reg config file

> configs/retro_small_model_wikitext103-gpt2-coeff0196-neigh.json

If you would like to retro-fit a trained retro-off checkpoint, specify `"true_retrofit" : "True"` in the config file (this is called true_retrofit as opposed to how we retro-fitted gpt-2 blocks). This will in the case of retro-on add CCA blocks and freeze everything else, in the case of retro-off not change anything.
If you would like to finetune on the validation set, so-called "minifit", set `"minifit" : "True"` in the config file, this will ignore the train dataset filepath and simply sample 15% of the validation dataset to train on, while keeping the other 85% to validate on.

## Eval.py
This script evaluates the given model checkpoint on a validation set. If you would like to load the same validation dataset this model was trained on, do not specify a validation dataset when you call this script.
This script is best explained via the options you have:
|Option| Explanation |
|--|--|
|checkpoint_path | Path to the model checkpoint for the model you would like to evaluate. no default.|
|val_dataset_path|Specify what dataset to evaluate on. If you leave this empty, it will evaluate on the same dataset we used to evaluate the model during training. Default: None|
|output_dir_path|Path to an output directory. This is the directory where you would save the perplexities and losses of each sample in the validation set in order to find intersting samples to generate for instance. If you leave this empty / do not specify, it will only print the result and not generate this output csv file. Default: ""|
|seventy_five_perc_window| Whether to employ the 75% sliding window approach, where you only evaluate the loss on the tokens for which the model already has 75% of the context window. Options: "True", "False" Default: "True"|
|shuffled_neighbors| This specifies if we shuffle the neighbors before we take the top-k for evaluation. Usually we evaluate on all 10 neighbors, in this case we wouldn't. **Note**: At the moment this only works for a batch size of 2. Options: "True", "False" Default: "False"|
|no_neighbors|Whether you would like to "turn off" the neighbors for Retro-li-on. If True, this would remove all "real" neighbors and replace it with a single neighbor per chunk. This neighbor will be the chunk itself and its continuation will be masked out. Options: "True", "False" Default: "False"|
|generate_manually_flag|This flag specifies if you want to generate next chunk based on a context of 75% for a certain validation sample. It would save the greedy output, the multinomial output and the top-p (p=0.9) output for your specified sample in a .txt file. Options: "True", "False" Default: "False"|
|generate_manually_index|This option goes with the one above. It specifies with sample to generate. For the generated outputs, you need to check the csv files created via"output_dir_path"! Options: Integer numbers Default: "-1"|
|outfile_name| This specifies where to save the generated samples. Default: None|
|noisy_flag| This flag specifies whether or not to add Gaussian noise to the neighbor embeddings at evaluation time. Options "True", "False" Default: "False"|
|noisy_coeff| This goes with the option above and specifies what noise coefficient to add to the Gaussian noise. Options: "0.196", "0.4", "1.0" Default: "0"|
### Running it
Running this on a GPU is fastest,  a typical example script would be:

    python eval.py --checkpoint_path "test/7c146192def711ee94cc00000c9dfe80/best-checkpoint-model.pt
 
### Acknowledgement
We reuse some of the infrastructure available in [labml.ai/labml](https://github.com/labmlai/labml) for building the models and training

#### Citation
```
@InProceedings{rashitig2024retroli,
    Author      = {Rashiti, Gentiana and Karunaratne, Geethan and Sachen, Mrinmaya and Sebastian, Abu and Rahimi, Abbas},
    Title       = {{Retro-Li: Small-Scale Retrieval Augmented Generation Supporting Noisy Similarity Searches and  Domain Shift Generalization}},
    Year        = {2024}
    Booktitle   = {Proceedings of the 27th European Conference on Artificial Intelligence},
}

```

#### License
Our code is licensed under MIT Licence. Please refer to the LICENSE file for the licensing of our code.