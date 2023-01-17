from nn import residual_layer
from pp import prep_dirs, data_aug


import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import datetime
import os


HOME = str(os.environ["WORK"])
HP = str(os.environ["HYPERPARAMS"])

SAVE_PATH = os.path.join(HOME, "ECGModels")
FIG_PATH = os.path.join(HOME, "figures")


if __name__ == "__main__":

    # --------------- set up --------------

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if not os.path.exists(FIG_PATH):
        os.mkdir(FIG_PATH)

    assert os.path.exists(SAVE_PATH) and os.path.exists(FIG_PATH), "paths must exist"

    # ------------- prep data ------------------
    train_mit, test_mit, tmp1, tmp2 = prep_dirs(HOME, HP)

    mit_data = pd.concat([train_mit, test_mit], axis=0)
    mit_data = pd.DataFrame(data_aug(mit_data))

    # ----------- create x and y --------------
    X = mit_data.iloc[:, :-1]
    y = mit_data.iloc[:, -1]

    # ------------y one hot matrix of 1 ------------
    y = to_categorical(y, num_classes=5)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True
    )

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    n_obs, n_feats, s_len = x_train.shape
    shape = (n_obs, n_feats, s_len)

    # ------- build model -------------

    # loss_fn = tf.keras.losses.CategoricalCrossentropy()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.75
    )
    lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    optimizer_obj = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999
    )

    inpus = tf.keras.Input(shape=(n_feats, s_len))
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1)(inpus)
    x1 = residual_layer()(x)
    x2 = residual_layer()(x1)
    x3 = residual_layer()(x2)
    x4 = residual_layer()(x3)
    x5 = residual_layer()(x4)
    x6 = tf.keras.layers.Flatten()(x5)
    x7 = tf.keras.layers.Dense(units=32, activation="relu")(x6)
    x8 = tf.keras.layers.Dense(units=32)(x7)
    x9 = tf.keras.layers.Dense(units=5)(x8)
    outpus = tf.keras.layers.Softmax()(x9)

    model = tf.keras.Model(inpus, outpus, name="ECGModel")

    # model = nn.ECGModel()

    model.compile(
        optimizer=optimizer_obj, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    start = datetime.datetime.now()

    # ---------- train model -------------

    num_epochs = 75

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=n_obs,
        verbose="2",
        validation_data=(x_test, y_test),
        callbacks=[lr],
    )

    today = datetime.datetime.today()
    today = today.isoformat().split("T")[0].replace("-", "_")

    # ------------ save model --------------------
    name = f"model_{today}"
    MODEL = os.path.join(SAVE_PATH, name)
    model.save(MODEL)

    duration = datetime.datetime.now() - start
    print("Training completed in time: ", duration)

    # --------------------- plotting -----------------
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(accuracy) + 1)  # include last

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax[0].plot(epochs, accuracy, "bo", label="Training accuracy")
    ax[0].plot(epochs, val_accuracy, "b", label="Validation accuracy")
    ax[0].set_title("Training and validation accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, loss, "bo", label="Training loss")
    ax[1].plot(epochs, val_loss, "b", label="Validation loss")
    ax[1].set_title("Training and validation loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.suptitle(f"ECG Model learning over {num_epochs} epochs")

    os.chdir(FIG_PATH)
    name = os.path.join(FIG_PATH, name)
    fig.savefig(
        name, dpi="figure", format="jpg", pad_inches=0.1, orientation="landscape"
    )
