"""
    pre action list belows

    !pip install efficientnet tensorflow_addons > /dev/null
    SAVEDIR = Path("models")
    SAVEDIR.mkdir(exist_ok=True)

    OOFDIR = Path("oof")
    OOFDIR.mkdir(exist_ok=True)
"""

import os
import math
import random
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.signal import get_window
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

#from config import CONFIG
from dataLoad import *
from Modeling import build_model, get_lr_callback

NUM_FOLDS = 4
IMAGE_SIZE = 256
BATCH_SIZE = 32
EFFICIENTNET_SIZE = 0
WEIGHTS = "imagenet"

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def auto_select_accelerator(): # TPU Setting
    TPU_DETECTED = False
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
        TPU_DETECTED = True
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    return strategy, TPU_DETECTED


set_seed(1213)
cfg = CONFIG()

strategy, tpu_detected = auto_select_accelerator()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync

all_files = get_datapath()

kf = KFold(n_splits=cfg['NUM_FOLDS'], shuffle=True, random_state=1213)
oof_pred = []
oof_target = []

files_train_all = np.array(all_files[:4])

for fold, (trn_idx, val_idx) in enumerate(kf.split(files_train_all)):
    files_train = files_train_all[trn_idx]
    files_valid = files_train_all[val_idx]

    print("=" * 120)
    print(f"Fold {fold}")
    print("=" * 120)

    train_image_count = count_data_items(files_train) # 왜 28000?
    valid_image_count = count_data_items(files_valid)

    tf.keras.backend.clear_session() # model의 복잡도로 올라간 memory 등을 초기화 함

    strategy, tpu_detected = auto_select_accelerator()
    with strategy.scope():
        model = build_model(
            size=IMAGE_SIZE, 
            efficientnet_size=EFFICIENTNET_SIZE,
            weights=WEIGHTS, 
            count=train_image_count // BATCH_SIZE // REPLICAS // 4)
    
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(SAVEDIR / f"fold{fold}.h5"), monitor="val_auc", verbose=1, save_best_only=True,
        save_weights_only=True, mode="max", save_freq="epoch"
    )

    history = model.fit(
        get_dataset(files_train, batch_size=BATCH_SIZE, shuffle=True, repeat=True, aug=True),
        epochs=EPOCHS,
        callbacks=[model_ckpt, get_lr_callback(BATCH_SIZE, REPLICAS)],
        steps_per_epoch=train_image_count // BATCH_SIZE // REPLICAS // 4,
        validation_data=get_dataset(files_valid, batch_size=BATCH_SIZE * 4, repeat=False, shuffle=False, aug=False),
        verbose=1
    )

    print("Loading best model...")
    model.load_weights(str(SAVEDIR / f"fold{fold}.h5"))

    ds_valid = get_dataset(files_valid, labeled=False, return_image_ids=False, repeat=True, shuffle=False, batch_size=BATCH_SIZE * 2, aug=False)
    STEPS = valid_image_count / BATCH_SIZE / 2 / REPLICAS
    pred = model.predict(ds_valid, steps=STEPS, verbose=0)[:valid_image_count]
    oof_pred.append(np.mean(pred.reshape((valid_image_count, 1), order="F"), axis=1))

    ds_valid = get_dataset(files_valid, repeat=False, labeled=True, return_image_ids=True, aug=False)
    oof_target.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))

    plt.figure(figsize=(8, 6))
    sns.distplot(oof_pred[-1])
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(
        np.arange(len(history.history["auc"])),
        history.history["auc"],
        "-o",
        label="Train auc",
        color="#ff7f0e")
    plt.plot(
        np.arange(len(history.history["auc"])),
        history.history["val_auc"],
        "-o",
        label="Val auc",
        color="#1f77b4")
    
    x = np.argmax(history.history["val_auc"])
    y = np.max(history.history["val_auc"])

    xdist = plt.xlim()[1] - plt.xlim()[0]
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color="#1f77b4")
    plt.text(x - 0.03 * xdist, y - 0.13 * ydist, f"max auc\n{y}", size=14)

    plt.ylabel("auc", size=14)
    plt.xlabel("Epoch", size=14)
    plt.legend(loc=2)

    plt2 = plt.gca().twinx()
    plt2.plot(
        np.arange(len(history.history["auc"])),
        history.history["loss"],
        "-o",
        label="Train Loss",
        color="#2ca02c")
    plt2.plot(
        np.arange(len(history.history["auc"])),
        history.history["val_loss"],
        "-o",
        label="Val Loss",
        color="#d62728")
    
    x = np.argmin(history.history["val_loss"])
    y = np.min(history.history["val_loss"])
    
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color="#d62728")
    plt.text(x - 0.03 * xdist, y + 0.05 * ydist, "min loss", size=14)

    plt.ylabel("Loss", size=14)
    plt.title(f"Fold {fold + 1} - Image Size {IMAGE_SIZE}, EfficientNetB{EFFICIENTNET_SIZE}", size=18)

    plt.legend(loc=3)
    plt.savefig(OOFDIR / f"fig{fold}.png")
    plt.show()