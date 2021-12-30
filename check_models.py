import tensorflow as tf
import argparse
import os
import json
import wandb
import logging
import pytest
# Contrived example of generating a module named as a string

from datetime import datetime
from wandb.keras import WandbCallback
from dataset import ImageNet
from utils import *
from dacite import from_dict

NORMALIZED = False


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
    label_smoothing=0.1,
    lr_schedule="half_cos",
    log_dir=log_location + "/logs",
    model_dir=log_location + "/models",
)


train_prep_cfg = get_preprocessing_config(
    tfrecs_filepath=train_tfrecs_filepath,
    batch_size=1024,
    image_size=512,
    area_factor=0.25,
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
all_weights = """
y002,70.3,68.51,1.79,gs://ak-us-train/models/11_16_2021_03h54m11s/all_model_epoch_94
y006,75.5,73.52,1.98,gs://ak-us-train/models/12_05_2021_13h26m36s/all_model_epoch_94
y008,76.3,74.48,1.82,gs://ak-us-train/models/12_01_2021_07h24m52s/all_model_epoch_94
y016,77.9,76.95,0.95,gs://ak-us-train/models/11_28_2021_10h11m52s/all_model_epoch_94
y032,78.9,78.05,0.85,gs://ak-us-train/models/11_27_2021_03h48m42s/all_model_epoch_94
y040,79.4,78.20,1.20,gs://ak-us-train/models/11_26_2021_11h30m09s/all_model_epoch_95
y064,79.9,78.95,0.95,gs://ak-us-train/models/11_26_2021_03h36m02s/all_model_epoch_95
y080,79.9,79.11,0.69,gs://ak-us-train/models/11_24_2021_09h10m48s/all_model_epoch_96
y120,80.3,79.45,0.85,gs://ak-us-train/models/11_22_2021_08h43m05s/all_model_epoch_96
y160,80.4,79.71,0.69,gs://ak-us-train/models/11_23_2021_13h33m06s/all_model_epoch_95
y320,80.9,80.12,0.78,gs://ak-us-train/models/11_26_2021_03h14m49s/all_model_epoch_96
"""
all_weights = all_weights.split("\n")
all_weights.remove('')
all_weights.remove('')
all_weights = list(map(lambda x: x.split(","),all_weights))

names_to_classes = {
    "y002" : tf.keras.applications.RegNetY002,
    "y004" : tf.keras.applications.RegNetY004,
    "y006" : tf.keras.applications.RegNetY006,
    "y008" : tf.keras.applications.RegNetY008,
    "y016" : tf.keras.applications.RegNetY016,
    "y032" : tf.keras.applications.RegNetY032,
    "y040" : tf.keras.applications.RegNetY040,
    "y064" : tf.keras.applications.RegNetY064,
    "y080" : tf.keras.applications.RegNetY080,
    "y120" : tf.keras.applications.RegNetY120,
    "y160" : tf.keras.applications.RegNetY160,
    "y320" : tf.keras.applications.RegNetY320
}

weights = dict()

for entry in all_weights:
    weights[entry[0]] = {
        "paper_acc": float(entry[1]),
        "actual_acc": float(entry[2]),
        "diff": float(entry[3]),
        "path": entry[4]
    }
val_ds = ImageNet(val_prep_cfg).make_dataset()
for i in weights:
    print(i)
    
import os

cluster_resolver, strategy = connect_to_tpu()

optim = get_optimizer(train_cfg)
for weight in weights:
    
    with strategy.scope():
        model = names_to_classes[weight]()
        model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=train_cfg.label_smoothing),
        optimizer="adam",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],)
#         model.load_weights(weights[weight]["path"])

        avg_loss = 0
        avg_acc = 0
        avg_top5 = 0

        for _ in range(10):

            metrics = model.evaluate(val_ds, steps=50, verbose=1)
            avg_loss += metrics[0]
            avg_acc += metrics[1]
            avg_top5 += metrics[2]
        
        print("Avg loss: ", avg_loss/10.)
        print("Avg acc: ", avg_acc/10.)
        print("Avg top5: ", avg_top5/10.)
        
        print("Assertion for ", weight)
        
        assert avg_acc/10. == pytest.approx(weights[weight]["actual_acc"] / 100., 0.001)
    savepath = "./"
    variant=weight
    with_head_savepath = os.path.join(savepath, "regnet" + variant + ".h5")
    without_head_savepath = os.path.join(savepath, "regnet" + variant + "_notop.h5")
    print("With head savepath:",with_head_savepath )
    print("Without head savepath", without_head_savepath )
    model.save(with_head_savepath,  include_optimizer=False)
    headless_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    headless_model.save(without_head_savepath)
