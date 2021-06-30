import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers

'''
用tf神经网络框架进行分类，使用了更多技巧：早停，drop-out等
'''

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
sam = pd.read_csv('sample_submission.csv')

train = df1.copy()
test = df2.copy()
submission = sam.set_index('id')
targets = pd.get_dummies(df1['target'])

cce = tf.keras.losses.CategoricalCrossentropy()


def custom_metric(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = K.mean(cce(y_true, y_pred))
    return loss


es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-05,
    patience=8,
    verbose=0,
    mode='min',
    baseline=None,
    restore_best_weights=True,
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=2,
    verbose=0,
)


def get_model():
    # 输入层
    inputs = layers.Input(shape=(75, ))

    # 嵌入层
    embed = layers.Embedding(360, 8)(inputs)
    embed = layers.Flatten()(embed)

    # 隐藏层
    hidden = layers.Dropout(0.2)(embed)
    hidden = tfa.layers.WeightNormalization(
        layers.Dense(
            units=48,
            activation='selu',
            kernel_initializer="lecun_normal",
        ))(hidden)

    # 输出层
    output = layers.Dropout(0.2)(layers.Concatenate()([embed, hidden]))
    output = tfa.layers.WeightNormalization(
        layers.Dense(units=48, activation='relu'))(output)

    output = layers.Dropout(0.3)(layers.Concatenate()([embed, hidden, output]))
    output = tfa.layers.WeightNormalization(
        layers.Dense(units=48, activation='elu'))(output)
    output = layers.Dense(9, activation='softmax')(output)

    model = keras.Model(inputs=inputs, outputs=output, name="res_nn_model")

    return model


EPOCH = 50
SEED = 123
N_FOLDS = 10

NN_a_train_preds = []
NN_a_test_preds = []
NN_a_oof_pred3 = []

oof_NN_a = np.zeros((train.shape[0], 9))
pred_NN_a = np.zeros((test.shape[0], 9))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_idx, ts_idx) in enumerate(skf.split(train, train.iloc[:, -1])):

    X_train = train.iloc[:, 1:-1].iloc[tr_idx]
    y_train = targets.iloc[tr_idx]
    X_test = train.iloc[:, 1:-1].iloc[ts_idx]
    y_test = targets.iloc[ts_idx]
    K.clear_session()

    model_attention = get_model()

    model_attention.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        metrics=custom_metric)

    model_attention.fit(
        X_train,
        y_train,
        batch_size=1024,
        epochs=EPOCH,
        validation_data=(X_test, y_test),
        callbacks=[es, plateau],
        verbose=0,
    )

    pred_a = model_attention.predict(X_test)
    oof_NN_a[ts_idx] += pred_a
    score_NN_a = log_loss(y_test, pred_a)
    print(f"\nFOLD {fold} Score NN Attention model: {score_NN_a}")
    pred_NN_a += model_attention.predict(test.iloc[:, 1:]) / N_FOLDS

    NN_a_train_preds.append(oof_NN_a[ts_idx])
    NN_a_test_preds.append(model_attention.predict(test.iloc[:, 1:]))

pred3 = sum(np.array(NN_a_test_preds) / N_FOLDS)

sub3 = sam.copy()
sub3.iloc[:, 1:] = pred3.data
sub3.to_csv("tf_classify.csv", index=False)
