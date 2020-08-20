import logging
import os

import numpy as np
import torch as T
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from pytorch_lightning import LightningModule

from .utils import convert_examples_to_features, get_labels, read_examples_from_file

logger = logging.getLogger(__name__)


class NERTransformer1(LightningModule):
    def __init__(self, hparams, **config_kwargs):
        super().__init__()

        cache_dir = "./cache"

        self.labels = ["TERM", "O"]
        num_labels = len(self.labels)

        self.config = AutoConfig.from_pretrained(
            hparams.model_name_or_path,
            **{"num_labels": num_labels},
            cache_dir=cache_dir,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            hparams.model_name_or_path, config=self.config, cache_dir=cache_dir,
        )

    def forward(self, **inputs):
        return self.model(**inputs)


class NERTransformer(LightningModule):
    def __init__(self, hparams, **config_kwargs):
        super().__init__()

        self.hparams = hparams

        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.labels = []
        self.labels = get_labels(hparams.labels)
        num_labels = len(self.labels)

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name,
            **{"num_labels": num_labels},
            cache_dir=cache_dir,
            **config_kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name, cache_dir=cache_dir,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.hparams.model_name_or_path, config=self.config, cache_dir=cache_dir,
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        tqdm_dict = {
            "loss": "{:.3f}".format(avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        return tqdm_dict

    def training_step(self, batch, batch_num):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}

        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        ret, _, _ = self._eval_end(outputs)
        logs = ret["log"]
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def validation_epoch_end(self, outputs):
        ret, _, _ = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_nb):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3],
        }

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {
            "val_loss": tmp_eval_loss.detach().cpu(),
            "pred": preds,
            "target": out_label_ids,
        }

    def _eval_end(self, outputs):
        val_loss_mean = T.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        label_map = {i: label for i, label in enumerate(self.labels)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "val_loss": val_loss_mean,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams

        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)

            if not os.path.exists(cached_features_file):
                logger.info("Creating features from dataset file at %s", args.data_dir)

                examples = read_examples_from_file(args.data_dir, mode)
                features = convert_examples_to_features(
                    examples,
                    self.labels,
                    args.max_seq_length,
                    self.tokenizer,
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=0,
                    sep_token=self.tokenizer.sep_token,
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info("Saving features into cached file %s", cached_features_file)

                T.save(features, cached_features_file)

    def load_dataset(self, mode, batch_size):
        "Load datasets. Called after prepare data."

        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)

        features = T.load(cached_features_file)
        all_input_ids = T.tensor([f.input_ids for f in features], dtype=T.long)
        all_attention_mask = T.tensor(
            [f.attention_mask for f in features], dtype=T.long
        )

        if features[0].token_type_ids is not None:
            all_token_type_ids = T.tensor(
                [f.token_type_ids for f in features], dtype=T.long
            )
        else:
            all_token_type_ids = T.tensor([0 for f in features], dtype=T.long)
            # HACK(we will not use this anymore soon)

        all_label_ids = T.tensor([f.label_ids for f in features], dtype=T.long)

        return DataLoader(
            TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids
            ),
            batch_size=batch_size,
        )

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size)

        t_total = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return self.load_dataset("dev", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.load_dataset("test", self.hparams.eval_batch_size)

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )
