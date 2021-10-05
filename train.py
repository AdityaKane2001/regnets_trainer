"""Script for training RegNetY. Supports TPU training."""

import tensorflow as tf
import argparse
import os
import json
import wandb
import logging
# Contrived example of generating a module named as a string

from datetime import datetime
from wandb.keras import WandbCallback
from dataset import ImageNet
from utils import *



NORMALIZED = False


log_location = "gs://ak-us-train"
train_tfrecs_filepath = tf.io.gfile.glob(
    "gs://adityakane-imagenet-tfrecs/train_*.tfrecord")
val_tfrecs_filepath = tf.io.gfile.glob(
    "gs://adityakane-imagenet-tfrecs/valid_*.tfrecord")

logging.basicConfig(format="%(asctime)s %(levelname)s : %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)

cluster_resolver, strategy = connect_to_tpu()

train_cfg = get_train_config(
    optimizer="sgd",
    base_lr=0.1 * strategy.num_replicas_in_sync,
    warmup_epochs=5,
    warmup_factor=0.1,
    total_epochs=100,
    weight_decay=5e-5,
    momentum=0.9,
    label_smoothing=0.0,
    lr_schedule="half_cos",
    log_dir=log_location + "/logs",
    model_dir=log_location + "/models",
)


train_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=train_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    crop_size=224,
    resize_pre_crop=320,
    augment_fn="default",
    num_classes=1000,
    color_jitter=True,
    mixup=False,
)

val_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=val_tfrecs_filepath,
    augment_fn="val",
    mixup=False
)

logging.info(f"Training options detected: {train_cfg}")
logging.info("Preprocessing options detected.")
logging.info(
    f"Training on TFRecords: {train_prep_cfg.tfrecs_filepath[0]} to {train_prep_cfg.tfrecs_filepath[-1]}")
logging.info(
    f"Validating on TFRecords: {val_prep_cfg.tfrecs_filepath[0]} to {val_prep_cfg.tfrecs_filepath[-1]}")

with strategy.scope():
    optim = get_optimizer(train_cfg)
    ## Enter model here
    model = tf.keras.applications.RegNetX002()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=train_cfg.label_smoothing),
        optimizer=optim,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # model.load_weights(log_location + "/init_weights/" + flops.upper())
    logging.info("Model loaded")

train_ds = ImageNet(train_prep_cfg).make_dataset()
val_ds = ImageNet(val_prep_cfg).make_dataset()

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm")

misc_dict = {
    "Rescaling": "1/255",
    "Normalization": "None"
}
config_dict = get_config_dict(train_prep_cfg, val_prep_cfg, train_cfg, misc_dict=misc_dict)

logging.info(config_dict)

wandb.init(entity="compyle", project="keras-regnet-training",
           job_type="train",  name=model.name + "_" + date_time,
           config=config_dict)


callbacks = get_callbacks(train_cfg, date_time)

history = model.fit(
    train_ds,
   	epochs=train_cfg.total_epochs,
   	validation_data=val_ds,
   	callbacks=callbacks,
)

with tf.io.gfile.GFile(os.path.join(train_cfg.log_dir, "history_%s.json" % date_time), "a+") as f:
   json.dump(str(history.history), f)
