from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset
import os
import random
import torch
from typing import Dict
import json

logger = logging.get_logger(__name__)


class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    adapted from https://github.com/huggingface/transformers/blob/91ff480e2693f36b11aaebc4e9cc79e4e3c049da/src/
    transformers/data/datasets/language_modeling.py
    """

    def __init__(self, tokenizer: PreTrainedTokenizer = None, file_dir: str = '', block_size: int = -100,
                 short_seq_prob=0.1, load_from_path=None):

        self.examples = []

        if load_from_path is not None:
            self.examples = self._load_from_path(load_from_path)
            return

        self.file_dir = file_dir
        self.block_size = block_size
        self.short_seq_prob = short_seq_prob
        self.tokenizer_path = tokenizer.name_or_path

        assert tokenizer is not None
        assert block_size != -100
        assert os.path.isdir(file_dir)
        logger.info(f"Creating features from dataset file folder at {file_dir}")

        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # file path looks like ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            with open(file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
                article_lines = []
                for i, line in enumerate(original_lines):
                    if (line.strip() == '' or i == len(original_lines) - 1) and len(article_lines) > 0:
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines
                            if (len(line) > 0 and not line.isspace())
                        ]

                        examples = self.create_examples_from_document(document, block_size, tokenizer, short_seq_prob)
                        self.examples.extend(examples)
                        article_lines = []
                    else:
                        article_lines.append(line)

        logger.info("Dataset parse finished.")

    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=0.1):
        """Creates examples for a single document.
        seems like hugging face took it from Google's ALBERT Repo:
        https://github.com/google-research/albert/blob/master/create_pretraining_data.py"""

        # Account for special tokens ([CLS], [SEP], [SEP])
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        examples = []
        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]  # get a segment
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)  # add a segment to current chunk
            current_length += len(segment)  # overall token length
            # if current length goes to the target length or reaches the end of file, start building token a and b
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
                    a_end = 1
                    # if current chunk has more than 2 sentences, pick part of it `A` (first) sentence
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    # token a
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # token b
                    tokens_b = []
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if len(tokens_a) == 0 or len(tokens_b) == 0:
                        continue

                    # switch tokens_a and tokens_b randomly
                    if random.random() < 0.5:
                        is_next = False
                        tokens_a, tokens_b = tokens_b, tokens_a
                    else:
                        is_next = True

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            assert len(trunc_tokens) >= 1
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "sentence_order_label": torch.tensor(0 if is_next else 1, dtype=torch.long),
                    }
                    examples.append(example)
                current_chunk = []  # clear current chunk
                current_length = 0  # reset current text length
            i += 1  # go to next line
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

    def save_to_disk(self, dataset_path: str):

        self._save_examples(self.examples, dataset_path)

        logger.info("Dataset saved in {}".format(dataset_path))

    def _save_examples(self, examples, dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        # get dataset info
        info_dict = {}

        info_dict['tokenizer_path'] = self.tokenizer_path
        info_dict['file_dir'] = self.file_dir
        info_dict['block_size'] = self.block_size
        info_dict['short_seq_prob'] = self.short_seq_prob
        info_dict['dataset_name'] = 'LineByLineWithSOPTextDataset'
        json.dump(info_dict, open(f'{dataset_path}/info.json', 'w'), indent=4, sort_keys=True)

        ids_tensor = []
        token_type_ids_tensor = []
        sentence_order_label_tensor = []

        for example in examples:
            ids_tensor.append(example['input_ids'])
            token_type_ids_tensor.append(example['token_type_ids'])
            sentence_order_label_tensor.append(example['sentence_order_label'])

        torch.save(ids_tensor, f'{dataset_path}/ids.tensor')
        torch.save(token_type_ids_tensor, f'{dataset_path}/token_type_ids.tensor')
        torch.save(sentence_order_label_tensor, f'{dataset_path}/sop_labels.tensor')

    def save_train_eval_splits(self, dataset_path: str, eval_p=0.05):
        # we do not shuffle because we do not want to mix one document in different splits

        total_len = len(self.examples)
        train_len = int(total_len * (1 - eval_p))

        examples_train = self.examples[:train_len]
        examples_eval = self.examples[train_len:]

        self._save_examples(examples_train, dataset_path + '/train')
        self._save_examples(examples_eval, dataset_path + '/eval')

        logger.info("Datasets saved in {}".format(dataset_path))

    def _load_from_path(self, path):
        ids_tensor = torch.load(f'{path}/ids.tensor')
        token_type_ids_tensor = torch.load(f'{path}/token_type_ids.tensor')
        sentence_order_label_tensor = torch.load(f'{path}/sop_labels.tensor')

        examples = []
        for i in range(len(ids_tensor)):
            example = {
                "input_ids": ids_tensor[i],
                "token_type_ids": token_type_ids_tensor[i],
                "sentence_order_label": sentence_order_label_tensor[i]
            }
            examples.append(example)
        info_dict = json.load(open(f'{path}/info.json'))
        self.tokenizer_path = info_dict['tokenizer_path']
        self.file_dir = info_dict['file_dir']
        self.block_size = info_dict['block_size']
        self.short_seq_prob = info_dict['short_seq_prob']

        return examples