import glob
import os
from types import SimpleNamespace

import torch as T
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from .model import NERTransformer
from .utils import read_eval_features


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
    args.num_train_epochs = 3
    args.train_batch_size = 16
    args.eval_batch_size = 16
    args.n_gpu = 1
    args.gradient_accumulation_steps = 1
    args.max_grad_norm = 1
    args.weight_decay = 0.0
    args.learning_rate = 5e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    args.save_steps = 750

    model = NERTransformer(args)

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

    pad_token_label_id = CrossEntropyLoss().ignore_index

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
        pad_token_label_id=pad_token_label_id,
    )

    features = features[0:16]

    all_input_ids = T.tensor([f.input_ids for f in features], dtype=T.long)
    all_attention_mask = T.tensor([f.attention_mask for f in features], dtype=T.long)
    all_token_type_ids = T.tensor([f.token_type_ids for f in features], dtype=T.long)

    with T.no_grad():
        output = model.forward(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            token_type_ids=all_token_type_ids,
        )

        print(*output)
