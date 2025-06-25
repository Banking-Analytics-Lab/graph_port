import logging
import datetime

import numpy as np
import tensorflow as tf

from spektral.layers import (
    ECCConv,
    GATConv,
    GraphMasking,
)

from spektral.data import BatchLoader
from spektral.models.gcn import GCN

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from graph_port.utils import CustomSharpe
import graph_port.data_processing as dp
from graph_port.models import AbsModel
from graph_port.gnn_layers import ImportanceLayer
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import Progbar


log = logging.getLogger("__main__")


def build_gnn_model(F, S, n_out):
    initializer = tf.keras.initializers.GlorotNormal()
    l2_reg = 2.5e-4  # L2 regularization rate
    x_in = Input(shape=(None, F))
    a_in = Input(shape=(None, None))
    e_in = Input(shape=(None, None, None, 1))

    i = ECCConv(32, activation="relu")([x_in, a_in, e_in])
    # i = ECCConv(64, activation="relu")([i, a_in, e_in])
    # i = tf.keras.layers.Dropout(0.2)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    # i = ECCConv(32, activation="relu")([i, a_in, e_in])
    i = tf.keras.layers.Dropout(0.2)(i)

    # i = GATConv(8,attn_heads=8,concat_heads=False,activation="relu",kernel_regularizer=l2(l2_reg),attn_kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg),)([i, a_in])
    # i = tf.keras.layers.Dropout(0.2)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    i = ECCConv(32, activation="relu", kernel_initializer=initializer)([i, a_in, e_in])
    i = tf.keras.layers.Dropout(0.2)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    output = tf.keras.layers.Dense(n_out, activation="softmax")(i)

    # Build model
    model = Model(inputs=[x_in, a_in, e_in], outputs=output)
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss=CustomSharpe())

    return model


def build_gn_model(F, N, n_out):
    initializer = tf.keras.initializers.GlorotNormal(seed=42)
    l2_reg = 2.5e-4  # L2 regularization rate
    x_in = Input(shape=(None, F))
    a_in = Input(shape=(None, None))
    # e_in = Input(shape=(None, None, None, 1))

    # i = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x_in)
    i = GraphMasking()(x_in)
    i = GATConv(
        64,
        attn_heads=4,
        concat_heads=True,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        attn_kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
    )([i, a_in])
    # i = tf.keras.layers.Dropout(0.1)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    # i = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(i)
    # i = tf.keras.layers.Dropout(0.2)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    # i = GATConv(128,attn_heads=8,concat_heads=True,kernel_regularizer=l2(l2_reg),attn_kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))([i, a_in])
    i = tf.keras.layers.Dropout(0.2)(i)
    i = tf.keras.layers.BatchNormalization()(i)
    # i = GATConv(64,attn_heads=8,concat_heads=True,activation="relu",kernel_regularizer=l2(l2_reg),attn_kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))([i, a_in])
    i  # = tf.keras.layers.Dropout(0.2)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    output = tf.keras.layers.Dense(n_out, activation="softmax")(i)

    # Build model
    model = Model(inputs=[x_in, a_in], outputs=output)

    return model


class GN_Model(Model):
    def __init__(self, n_out=0, mask=True):
        super().__init__()

        l2_reg = 2.5e-4  # L2 regularization rate
        self.mask = mask
        self.masking = GraphMasking()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.eccconv = ECCConv(
            72,
            activation="relu",
        )  # 84  # 64
        self.gatconv1 = GATConv(
            24,  # 12  # 24,  # 10
            attn_heads=8,  # 8 # 10,  # 8
            concat_heads=True,
            add_self_loops=False,  # False
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            return_attn_coef=True,
            # dtype="float16",
        )
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.attn = None
        # self.dense1 = tf.keras.layers.Dense(32, activation="relu")  # 32
        # self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense2 = tf.keras.layers.Dense(16, activation="relu")  # 16
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(1, activity_regularizer=l1(2.5e-2))
        self.activation = tf.keras.layers.Activation("relu")
        self.imp = ImportanceLayer(activation="reduce_weights")

    def call(self, inputs, training=True):
        x, a, e = inputs
        # print(x.shape, a.shape, e.shape, self.mask)
        if self.mask:
            x = self.masking(x)

        x = self.batchnorm1(x, training=training)
        # x = self.eccconv([x, a, e])
        x, attn = self.gatconv1([x, a])
        x = self.dropout1(x, training=training)
        # x, attn = self.gatconv2([x, a])
        # x = self.dense1(x)
        # x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.dropout3(x, training=training)
        x = self.dense3(x)
        x = self.activation(x)
        output = self.imp([x, a])

        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = (
            tf.keras.Input(shape=(1, 6830, 4)),
            tf.keras.Input(shape=(1, 6830, 6830)),
        )
        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    def model_summary(self, inputs_shape):
        if len(inputs_shape) == 2:
            x = (
                tf.keras.Input(shape=inputs_shape[0]),
                tf.keras.Input(shape=inputs_shape[1]),
            )
        else:
            x = (
                tf.keras.Input(shape=inputs_shape[0]),
                tf.keras.Input(shape=inputs_shape[1]),
                tf.keras.Input(shape=inputs_shape[2]),
            )

        return Model(inputs=[x], outputs=self.call(x)).summary()


class GNNModel(AbsModel):
    def __init__(self, cfg, model=None):
        super().__init__(cfg)

        self.data = dp.get_data_from_path(cfg)
        self.model = model
        self.dataset = None
        self.loss_fn = CustomSharpe()
        self.optimizer = Adam(lr=1e-4)
        self.best_weights = None
        log.info("**Done Loading data**")

    def build_model(self):
        F = (
            self.data.n_node_features
        )  # + 1  # Dimension of node features# resolves some error in getting features, when masking
        S = self.data.n_edge_features  # Dimension of edge features

        n_out = self.data.n_labels  # Dimension of the target
        print(F, S, n_out)

        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        if self.data.n_max is None:
            self.model = GN_Model(n_out, mask=False)
            print(
                self.model.model_summary(
                    [[n_out, F], [n_out, n_out], [n_out, n_out, S]]
                )
            )

            # self.model = build_gnn_model(F, S, n_out)
        else:
            self.model = GN_Model(n_out, mask=True)
            print(
                self.model.model_summary(
                    [[n_out, F + 1], [n_out, n_out], [n_out, n_out, S]]
                )
            )

        optimizer = Adam(lr=5e-4)
        self.model.compile(optimizer=optimizer, loss=CustomSharpe())

    def prepare_train_val_test(self, batch_size):
        split_va, split_te = int(0.70 * len(self.data)), int(0.85 * len(self.data))
        idx_tr, idx_va, idx_te = np.split(range(len(self.data)), [split_va, split_te])
        dataset_tr = self.data[idx_tr]
        dataset_val = self.data[idx_va]
        dataset_test = self.data[idx_te]
        if self.data.n_max is not None:
            loader_tr = dp.MaxPackedBatchLoader(
                dataset_tr,
                self.data.n_max,
                mask=True,
                batch_size=batch_size,
                shuffle=False,
            )
            loader_val = dp.MaxPackedBatchLoader(
                dataset_val,
                self.data.n_max,
                mask=True,
                batch_size=batch_size,
                shuffle=False,
            )
            loader_test = dp.MaxPackedBatchLoader(
                dataset_test,
                self.data.n_max,
                mask=True,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, shuffle=False)
            loader_val = BatchLoader(dataset_val, batch_size=batch_size, shuffle=False)
            loader_test = BatchLoader(
                dataset_test, batch_size=batch_size, shuffle=False
            )

        self.dataset = [loader_tr, loader_val, loader_test]
        # return loader_tr, loader_val, loader_test

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        return self.loss_fn(y, val_logits)

    def train_model(self, batch_size, epochs, patience):
        if not self.model:
            self.build_model()
        print("**Training model started**")
        if self.dataset is None:
            self.prepare_train_val_test(batch_size)

        loader_tr, loader_val, loader_test = (
            self.dataset[0],
            self.dataset[1],
            self.dataset[2],
        )

        # metric_names = [
        #     "loss",
        # ]
        # wait = 0
        # best = 0
        #

        # for epoch in range(epochs):
        #     print("\nepoch {}/{}".format(epoch + 1, epochs))
        #     start_time = time.time()

        #     # Iterate over the batches of the dataset.
        #     pb_i = Progbar(loader_tr.steps_per_epoch, stateful_metrics=metric_names)

        #     for step, (x_batch_train, y_batch_train) in enumerate(loader_tr):
        #         loss_value = self.train_step(x_batch_train, y_batch_train)
        #         pb_i.add(batch_size, values=[("loss", loss_value)])
        #         if step + 1 == loader_tr.steps_per_epoch:
        #             break

        #     # Run a validation loop at the end of each epoch.
        #     batch_loss = []
        #     val_loss = 0
        #     for step, (x_batch_val, y_batch_val) in enumerate(loader_val):
        #         bl = self.test_step(x_batch_val, y_batch_val)
        #         batch_loss.append(bl.numpy())
        #         if step == loader_val.steps_per_epoch:
        #             break

        #     val_loss = np.mean(batch_loss)
        #     print("val_loss: %.4f" % (float(val_loss)))
        #     print("Time taken: %.2fs" % (time.time() - start_time))

        #     wait += 1
        #     if val_loss < best:
        #         best = val_loss
        #         wait = 0
        #         self.best_weights = self.model.get_weights()
        #     if wait >= patience:
        #         print("Early stopping: ", val_loss)
        #         self.model.set_weights(self.best_weights)
        #         break

        # run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
        # runmeta = tf.compat.v1.RunMetadata()
        self.model.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=epochs,
            validation_data=loader_val.load(),
            validation_steps=loader_val.steps_per_epoch,
            verbose=1,
            callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
        )

        print("**Training model finished**")
        mtr, btr = self.test_pred(loader_tr)
        mval, bval = self.test_pred(loader_val)
        mte, bte = self.test_pred(loader_test)
        print(
            "Sharpe Ratio Training: {:.4f}, Validation: {:.4f}, Test: {:.4f}".format(
                -1 * mtr, -1 * mval, -1 * mte
            )
        )

    def test_pred(self, loader):
        model_loss = 0
        curr_batch = 0
        batch_loss = []
        for batch in loader:
            curr_batch += 1

            inputs, target = batch
            # print(inputs[0].shape,inputs[1].shape,target.shape)
            predictions = self.model(inputs, training=False)
            bl = self.loss_fn(target, predictions)
            batch_loss.append(bl.numpy())
            model_loss += bl.numpy()

            if curr_batch == loader.steps_per_epoch:
                break
        model_loss /= loader.steps_per_epoch
        return model_loss, batch_loss
