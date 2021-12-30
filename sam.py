"""Script for training RegNetY. Supports TPU training."""

import tensorflow as tf
import argparse
import os
import json
import wandb
import logging

from datetime import datetime
from wandb.keras import WandbCallback
from dataset import ImageNet
from utils import *
from dacite import from_dict

NORMALIZED = False

# credits: https://github.com/sayakpaul/Sharpness-Aware-Minimization-TensorFlow
class SAMModel(tf.keras.Model):
    def __init__(self, resnet_model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.resnet_model = resnet_model
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.resnet_model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.resnet_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.resnet_model(images)
            loss = self.compiled_loss(labels, predictions)    
        
        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
        
        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.resnet_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm



log_location = "gs://ak-us-train"
train_tfrecs_filepath = tf.io.gfile.glob(
    "gs://ak-imagenet-new/train-2/train_*.tfrecord")
val_tfrecs_filepath = tf.io.gfile.glob(
    "gs://ak-imagenet-new/valid-2/valid_*.tfrecord")

logging.basicConfig(format="%(asctime)s %(levelname)s : %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)

cluster_resolver, strategy = connect_to_tpu()

train_cfg = get_train_config(
    optimizer="adamw",
    base_lr=0.001 * strategy.num_replicas_in_sync,       #################################################change this!!
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
    area_factor=0.08,
    crop_size=224,
    resize_pre_crop=256,
    augment_fn="default",
    num_classes=1000,
    color_jitter=False,
    mixup=False,
    mixup_alpha=0.2
)

val_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=val_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    area_factor=0.25,
    crop_size=224,
    resize_pre_crop=256,
    augment_fn="val",
    num_classes=1000,
    color_jitter=False,
    mixup=False,
    mixup_alpha=0.0
)

misc_dict = {
    "Rescaling": "1/255",
    "Normalization": "None",
}

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%Hh%Mm%Ss")

config_dict = get_config_dict(
    train_prep_cfg, val_prep_cfg, train_cfg, misc=misc_dict)

logging.info(config_dict)

wandb.init(entity="compyle", project="keras-regnet-training",
           job_type="train",  name="regnety002" + "_" + date_time, #################################################change this!!

           config=config_dict)
# train_cfg = wandb.config.train_cfg
# train_cfg = from_dict(data_class=TrainConfig, data=train_cfg)
logging.info(f"Training options detected: {train_cfg}")
logging.info("Preprocessing options detected.")
logging.info(
    f"Training on TFRecords: {train_prep_cfg.tfrecs_filepath[0]} to {train_prep_cfg.tfrecs_filepath[-1]}")
logging.info(
    f"Validating on TFRecords: {val_prep_cfg.tfrecs_filepath[0]} to {val_prep_cfg.tfrecs_filepath[-1]}")

INIT_EPOCH = 0

with strategy.scope():
    optim = get_optimizer(train_cfg)

    model = tf.keras.applications.RegNetY002() #################################################change this!!
    model = SAMModel(model)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=train_cfg.label_smoothing),
        optimizer=optim,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    if INIT_EPOCH > 0:
        model.load_weights("gs://ak-us-train/models/11_11_2021_10h23m04s/all_model_epoch_"+f"{INIT_EPOCH:02}")
    logging.info("Model loaded")

train_ds = ImageNet(train_prep_cfg).make_dataset()
# train_ds = train_ds.shuffle(300)
val_ds = ImageNet(val_prep_cfg).make_dataset()
# val_ds = val_ds.shuffle(48)


callbacks = get_callbacks(train_cfg, date_time)
if INIT_EPOCH > 0:
    count = 1252*INIT_EPOCH

    for i in range(len(callbacks)):
        try:
            callbacks[i].count = count
        except:
            pass

history = model.fit(
    train_ds,
   	epochs=train_cfg.total_epochs,
    steps_per_epoch=1252,
   	validation_data=val_ds,
#     validation_steps=50,
   	callbacks=callbacks,
#     steps_per_epoch = 1251,
    validation_steps = 49,
    initial_epoch=INIT_EPOCH
)

with tf.io.gfile.GFile(os.path.join(train_cfg.log_dir, "history_%s.json" % date_time), "a+") as f:
   json.dump(str(history.history), f)
