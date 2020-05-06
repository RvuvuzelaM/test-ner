import glob
import os
from types import SimpleNamespace

from .model import NERTransformer
from .trainer import create_trainer


def train():
    args = SimpleNamespace()

    args.data_dir = './data'
    args.labels = './data/labels.txt'
    args.model_name_or_path = 'bert-base-multilingual-cased'
    args.config_name = 'bert-base-multilingual-cased'
    args.tokenizer_name = 'bert-base-multilingual-cased'
    args.output_dir = './germeval-model'
    args.cache_dir = './cache'

    args.max_seq_length = 32
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
    trainer = create_trainer(model, args)

    trainer.fit(model)

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
    model = model.load_from_checkpoint(checkpoints[-1])
    trainer.test(model)
