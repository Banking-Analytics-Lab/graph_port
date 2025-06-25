import logging

import numpy as np
import tensorflow as tf

from spektral.layers import ECCConv, GATConv, GraphMasking, GlobalSumPool
from spektral.data import BatchLoader
from spektral.models.gcn import GCN
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from graph_port.utils import CustomSharpe
import graph_port.data_processing as dp
from graph_port.models import AbsModel
from graph_port.gnn_layers import ImportanceLayer
from tensorflow.keras.regularizers import l2

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
    initializer = tf.keras.initializers.GlorotNormal()
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
    def __init__(self, n_out, mask=True):
        super().__init__()

        initializer = tf.keras.initializers.GlorotNormal()
        l2_reg = 2.5e-4  # L2 regularization rate
        self.mask = mask
        self.masking = GraphMasking()
        self.gatconv1 = GATConv(
            32,
            attn_heads=4,
            concat_heads=True,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            # dtype="float16",
        )
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.gatconv2 = GATConv(
            64,
            attn_heads=4,
            concat_heads=True,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            # dtype="float16",
        )

        self.gatconv3 = GATConv(
            16,
            attn_heads=4,
            concat_heads=True,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            # dtype="float16",
        )
        self.gatconv4 = GATConv(
            1,
            attn_heads=1,
            concat_heads=True,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            # dtype="float16",
        )
        self.imp = ImportanceLayer()
        # self.global_pool = GlobalSumPool()
        # self.dense = tf.keras.layers.Dense(n_out, activation="softmax")

    def call(self, inputs):
        x, a = inputs
        if self.mask:
            x = self.masking(x)

        x = self.gatconv1([x, a])
        x = self.dropout1(x)
        x = self.batch_norm(x)
        x = self.gatconv2([x, a])
        x = self.gatconv3([x, a])
        x = self.gatconv4([x, a])
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
        x = (
            tf.keras.Input(shape=inputs_shape[0]),
            tf.keras.Input(shape=inputs_shape[1]),
        )
        return Model(inputs=[x], outputs=self.call(x)).summary()


class GNNModel(AbsModel):
    def __init__(self, cfg, model=None):
        super().__init__(cfg)

        self.data = dp.get_data_from_path(cfg)
        self.model = model
        self.dataset = None
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
            self.model = build_gnn_model(F, S, n_out)
        else:
            self.model = GN_Model(n_out, mask=True)

        optimizer = Adam(lr=5e-3)
        self.model.compile(optimizer=optimizer, loss=CustomSharpe())

        # print(self.model.model_summary())

    def prepare_train_val_test(self, batch_size, train_index, test_index):
        dataset_tr = self.data[train_index]
        dataset_val = self.data[test_index]
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

        else:
            loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, shuffle=False)
            loader_val = BatchLoader(dataset_val, batch_size=batch_size, shuffle=False)

        return loader_tr, loader_val

    def train_model(self, batch_size, epochs, patience):
        if not self.model:
            self.build_model()
        print("**Training model started**")
        tscv = TimeSeriesSplit()
        i = 0
        for train_index, test_index in tscv.split(self.data):
            print("TRAIN:", train_index, "TEST:", test_index)
            loader_tr, loader_val = self.prepare_train_val_test(
                batch_size, train_index, test_index
            )
            history = self.model.fit(
                loader_tr.load(),
                steps_per_epoch=loader_tr.steps_per_epoch,
                epochs=epochs,
                validation_data=loader_val.load(),
                validation_steps=loader_val.steps_per_epoch,
                verbose=1,
                callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
            )

            print("**Training finished on split ", i)
            mtr, btr = self.test_pred(loader_tr)
            mval, bval = self.test_pred(loader_val)
            print(
                "Sharpe Ratio Training: {:.4f}, Test: {:.4f}".format(
                    -1 * mtr, -1 * mval
                )
            )
            i += 1

    def test_pred(self, loader):
        model_loss = 0
        curr_batch = 0
        batch_loss = []
        loss_fn = CustomSharpe()
        for batch in loader:
            curr_batch += 1

            inputs, target = batch
            # print(inputs[0].shape,inputs[1].shape,target.shape)
            predictions = self.model(inputs, training=False)
            bl = loss_fn(target, predictions)
            batch_loss.append(bl.numpy())
            model_loss += bl.numpy()

            if curr_batch == loader.steps_per_epoch:
                break
        model_loss /= loader.steps_per_epoch
        return model_loss, batch_loss
