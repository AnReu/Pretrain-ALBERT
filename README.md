# Pretrain-ALBERT
contains pre-processing and pre-training scripts for ALBERT


## Usage

### Pre-Processing
First, we need to pre-process the data. Since this might take some time, we want to perform this step before and save the pre-processed data so that we do not waste our GPU time on pre-processing that actually does not need a GPU.

Call:

```bash
python create_dataset.py example_data example_data_tokenized
```

This will create the SOP labels used during pre-training as well as tokenize the data and convert it to tensors. This step will take some time (several hours) depending on your setup. In the script currently the ALBERT tokenizer is used. If you want to train BERT or any other model, you need to change it in this script.

As input it takes a directory which potentially can contain multiple file. All these files will then be processed as input data. The script will output two datasets, one validation set and one training set. They will be created in the output_dir. It will also output an info.json which contains the parameters to generate the dataset such as the input datapath and the tokenizer.

The files inside the input directory need to be in the following format:
1. Each sentence is on a new line.
2. Different documents are separated by two newlines, i.e., an empty line between them.

A document could be a tweet, a wikipedia article, a book chapter, social media post, etc. An example of the format can be found in the directory `example_data`.

Call the script with your own data as follows:
```bash
python create_dataset.py <input_dir> <output_dir (optional)>
```

### Pre-training

Call:
```bash
python pretrain.py
```

`pretrain.py` will start the pretraining of ALBERT. It will automatically load the tokenizer that was used to create the dataset. The dataset along with other information such as which model to use are specified in the script itself. In order to change the model, make sure that the SOP labels for your model are actually called sentence_order_label otherwise the huggingface Trainer will throw an error. For example, BERT calls them next_sentence_label. Apart from this, you can simply use any huggingface model, that has a masked language model head and classification head.

The `tokenized_data_dir` needs to be the output dir of the `get_hf_dataset.py` above as it relies on the files outputted by the script.

If `use_comet` is provided as a flag the experiment will be logged to your comet.ml dashboard. After the experiment is done, it will upload all hyperparameters to the experiment as well. Note that you need comet_ml installed and the environment variables for your project and api key set before you run the script.

Options:

- Edit `pretrain.py` to change pretraining hyperparameters
- `python pretrain.py use_comet` to log to comet_ml (needs to be installed)