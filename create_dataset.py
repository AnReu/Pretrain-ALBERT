from sop_dataset import LineByLineWithSOPTextDataset
from transformers import AlbertTokenizerFast
from os import makedirs
import sys

dataset = LineByLineWithSOPTextDataset
model_path = 'albert-base-v2' # huggingface model path, e.g., 'albert-base-v2'
tokenizer = AlbertTokenizerFast.from_pretrained(model_path)

assert len(sys.argv) > 1

input_data_dir = sys.argv[1]
dataset_path = sys.argv[2] if len(sys.argv) > 2 else input_data_dir + '_tokenized'

makedirs(dataset_path, exist_ok=True)


tokenized_data = dataset(
    tokenizer=tokenizer,
    file_dir=input_data_dir,
    block_size=512,
)

tokenized_data.save_train_eval_splits(dataset_path, eval_p=0.05)