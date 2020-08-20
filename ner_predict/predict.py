import glob
import os
from types import SimpleNamespace

import torch as T
from scipy.special import softmax
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from .model import NERTransformer1
from .utils import read_eval_features, read_examples_from_file


def predict():
    args = SimpleNamespace()

    args.data_dir = "./data"
    args.labels = "./data/labels.txt"
    args.model_name_or_path = "bert-base-multilingual-cased"
    args.config_name = "bert-base-multilingual-cased"
    args.tokenizer_name = "bert-base-multilingual-cased"
    args.output_dir = "./model"
    args.cache_dir = "./cache"

    args.max_seq_length = 64
    args.max_grad_norm = 1
    args.weight_decay = 0.0
    args.learning_rate = 5e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    args.save_steps = 750

    model = NERTransformer1(args)

    checkpoints = list(
        sorted(
            glob.glob(
                os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True
            )
        )
    )
    model = model.load_from_checkpoint(checkpoints[-1])

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, cache_dir=args.cache_dir,
    )

    features = read_eval_features(
        data_dir="data",
        file_name="dev",
        max_seq_length=64,
        tokenizer=tokenizer,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
    )

    all_input_ids = T.tensor([f.input_ids for f in features], dtype=T.long)
    all_attention_mask = T.tensor([f.attention_mask for f in features], dtype=T.long)
    all_token_type_ids = T.tensor([f.token_type_ids for f in features], dtype=T.long)

    with T.no_grad():
        output = model.forward(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            token_type_ids=all_token_type_ids,
        )

    preds = output[0].numpy()

    with open("out.txt", mode="w") as f:
        for input_ids, pred in zip(all_input_ids, preds):
            words = tokenizer.convert_ids_to_tokens(input_ids)[1:]
            probs = pred[1:]

            sentence = ""
            prev_word = ""

            prev_is_term = 0
            is_term = ""

            for word, prob in zip(words, probs):
                prob = softmax(prob)

                if word == "[SEP]" or word == "[PAD]":
                    if prev_word:
                        is_term += str(prev_is_term) + " "
                        sentence += prev_word
                    break

                a = word.split("#")
                if len(a) > 1:
                    prev_is_term += prob[1]
                    prev_is_term /= 2
                    prev_word += a[-1]
                else:
                    if prev_word:
                        is_term += str(prev_is_term) + " "
                        prev_is_term = 0
                        sentence += prev_word + " "
                        prev_word = ""

                    prev_word += a[0]
                    prev_is_term += prob[1]

            f.write(sentence + "\n")
            f.write(is_term + "\n")
