import logging
import traceback
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


from graph_port.utils import CustomSharpe
import graph_port.data_processing as dp

from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

from tensorflow.keras.regularizers import l2
from spektral.data import BatchLoader

log = logging.getLogger("__main__")
np.set_printoptions(suppress=True)


def build_lstm_model(input_dim, output_dim, batch_size=None):

    initializer = tf.keras.initializers.GlorotNormal()

    x_i = tf.keras.Input(batch_shape=(batch_size, input_dim[0], input_dim[1]))
    i = tf.keras.layers.LSTM(units=512, return_sequences=False, activation="relu")(x_i)
    i = tf.keras.layers.Dropout(0.1)(i)
    i = tf.keras.layers.BatchNormalization()(i)
    # i = tf.keras.layers.LSTM(units=1024,return_sequences=True,activation='relu')(i)
    # i = tf.keras.layers.Dropout(0.1)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    # `i = tf.keras.layers.LSTM(units=256,return_sequences=False,activation='relu')(x_i)
    # i = tf.keras.layers.Dropout(0.1)(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    o = tf.keras.layers.Dense(output_dim, activation="softmax")(i)
    lstm = tf.keras.Model(inputs=[x_i], outputs=[o])

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=1e-4, momentum=0.05, nesterov=True
    )
    # optimizer = Adam()
    lstm.compile(optimizer=optimizer, loss=CustomSharpe())

    return lstm


class AbsModel:
    def __init__(self, cfg):
        self.model_name = cfg.app_params.model
        self.model_params = cfg.model_params
        self.data = None
        self.model = None
        self.data_path = cfg.app_params.data_path

    def build_model(self):
        pass

    def prepare_train_val_test(self, batch_size):
        X = self.data[0]
        y = self.data[1]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.15, shuffle=False
        )
        # print('Train: ',x_train.shape,y_train.shape,'Val: ',x_val.shape,y_val.shape,' Test: ',x_test.shape,y_test.shape)
        print(
            "Train: ",
            len(x_train),
            len(y_train),
            "Val: ",
            len(x_val),
            len(y_val),
            " Test: ",
            len(x_test),
            len(y_test),
        )
        train_dataset = zip(x_train, y_train)
        val_dataset = zip(x_val, y_val)
        test_dataset = zip(x_test, y_test)

        return train_dataset, val_dataset, test_dataset

    def train_model(self, batch_size, epochs, patience):

        loader_tr, loader_val, loader_test = self.prepare_train_val_test(batch_size)
        mtr, btr, feas_tr = self.test_pred(loader_tr)
        mval, bval, feas_val = self.test_pred(loader_val)
        mte, bte, feas_te = self.test_pred(loader_test)
        print(
            "Sharpe Ratio Training: {:.4f}, Validation: {:.4f}, Test: {:.4f}".format(
                -1 * mtr, -1 * mval, -1 * mte
            )
        )
        print(
            "Feasibility %: {:.2f}, Validation: {:.2f}, Test: {:.2f}".format(
                feas_tr, feas_val, feas_te
            )
        )

    def test_model(self, test_data):
        pass


class LSTMModel(AbsModel):
    def __init__(self, cfg, model=None):
        super().__init__(cfg)
        self.data = dp.get_data_from_path(cfg)
        # self.x_dim,self.y_dim
        self.model = model
        self.input_size = 150  # to limit the history of data
        print("**Done Loading data**")

    def build_model(self):
        input_dim = (self.input_size, self.data[0].shape[2])
        output_dim = self.data[1].shape[2]
        self.model = build_lstm_model(input_dim, output_dim)
        print(self.model.summary())

    # def prepare_train_val_test(self,batch_size):
    #     DATASET_SIZE=self.data.cardinality().numpy()
    #     train_size = int(0.7 * DATASET_SIZE)
    #     val_size = int(0.15 * DATASET_SIZE)
    #     test_size = int(0.15 * DATASET_SIZE)

    #     train_dataset = self.data.take(train_size)
    #     test_dataset = self.data.skip(train_size)
    #     val_dataset = test_dataset.skip(val_size)
    #     test_dataset = test_dataset.take(test_size)

    #     train_dataset = train_dataset.batch(batch_size)
    #     val_dataset = val_dataset.batch(batch_size)
    #     test_dataset = test_dataset.batch(batch_size)

    #     return train_dataset,val_dataset,test_dataset

    def prepare_train_val_test(self, batch_size):
        X = self.data[0][:, -1 * self.input_size :, :]
        y = self.data[1]
        print("X: ", X.shape, "y:", y.shape)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.15, shuffle=False
        )
        print(
            "Train: ",
            x_train.shape,
            y_train.shape,
            "Val: ",
            x_val.shape,
            y_val.shape,
            " Test: ",
            x_test.shape,
            y_test.shape,
        )
        # train_dataset = (x_train,y_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, val_dataset, test_dataset

    def train_model(self, batch_size, epochs, patience):
        if not self.model:
            self.build_model()
        print("**Preparing data for training**")
        loader_tr, loader_val, loader_test = self.prepare_train_val_test(batch_size)
        print("**Starting Traning**")
        self.model.fit(
            loader_tr,
            epochs=epochs,
            validation_data=loader_val,
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
        loss_fn = CustomSharpe()
        for inputs, target in loader:
            curr_batch += 1
            predictions = self.model(inputs, training=False)
            # print('pred: ',predictions.shape,'tg: ',target.shape)
            bl = loss_fn(target, predictions)
            batch_loss.append(bl.numpy())
            model_loss += bl.numpy()

        model_loss /= curr_batch
        return model_loss, batch_loss


class EffPorModel(AbsModel):
    def __init__(self, cfg, model=None):
        super().__init__(cfg)
        dataset = dp.get_data_from_path(cfg)
        self.data = dataset.inp_data
        self.model = model
        print("**Done Loading data**")

    def build_model(self):
        pass

    def test_pred(self, loader):
        model_loss = 0
        curr_batch = 0
        batch_loss = []
        loss_fn = CustomSharpe(expand_dims=True)
        loader_count = 0
        for inputs, target in loader:
            loader_count += 1
            print(inputs.shape)
            print(inputs)
            try:
                mu = mean_historical_return(inputs, returns_data=True)
                S = risk_models.sample_cov(inputs, returns_data=True)
                ef = EfficientFrontier(
                    mu,
                    S,
                    solver_options={
                        "max_iter": 20000,
                        "eps_abs": 0.001,
                        "eps_rel": 0.001,
                    },
                )  # verbose=True to see performance
                ef.add_objective(objective_functions.L2_reg)
                weights_ef = ef.max_sharpe(risk_free_rate=0)
                val = np.zeros((1, len(weights_ef)))
                for i, key in enumerate(weights_ef.keys()):
                    val[0, i] = weights_ef[key]

                target_exp = np.expand_dims(target, axis=0)
                bl = loss_fn(target_exp, val)
                bl = bl.numpy()
                batch_loss.append(bl)
                model_loss += bl
                curr_batch += 1

            except Exception as e:
                print(e.__doc__)
                print(str(e))
                print(traceback.print_exc())
                continue

        if curr_batch > 0:
            model_loss /= curr_batch
        return model_loss, batch_loss, curr_batch / loader_count * 100


class EquPorModel(AbsModel):
    def __init__(self, cfg, model=None):
        super().__init__(cfg)
        dataset = dp.get_data_from_path(cfg)
        self.data = dataset.inp_data
        self.n_max = dataset.n_max
        self.model = model
        print("**Done Loading data**")

    def build_model(self):
        pass

    def test_pred(self, loader):
        model_loss = 0
        curr_batch = 0
        batch_loss = []
        loss_fn = CustomSharpe(expand_dims=True, test=True, trading_year_days=4.0)
        loader_count = 0
        for inputs, target in loader:
            loader_count += 1
            if self.n_max is not None:
                wgt_arr = np.ones((1, inputs.shape[1]))
            else:
                wgt_arr = np.ones((1, inputs.shape[0]))

            wgt_arr = np.round(wgt_arr / np.sum(wgt_arr), decimals=6)

            target_exp = np.expand_dims(target, axis=0)
            print(inputs.shape, wgt_arr.shape)
            bl = loss_fn(target_exp, wgt_arr)
            bl = bl.numpy()
            batch_loss.append(bl)
            model_loss += bl
            curr_batch += 1

        if curr_batch > 0:
            model_loss /= curr_batch

        return model_loss, batch_loss, curr_batch / loader_count * 100


class NetworkModel(AbsModel):
    def __init__(self, cfg, model=None):
        super().__init__(cfg)

        self.data = dp.get_data_from_path(cfg)
        self.mask = False if cfg.app_params.n_max is None else True
        self.model = model
        self.dataset = None
        log.info("**Done Loading data**")

    def build_model(self):
        pass

    def prepare_train_val_test(self, batch_size):
        split_va, split_te = int(0.70 * len(self.data)), int(0.85 * len(self.data))
        idx_tr, idx_va, idx_te = np.split(range(len(self.data)), [split_va, split_te])
        dataset_tr = self.data[idx_tr]
        dataset_val = self.data[idx_va]
        dataset_test = self.data[idx_te]
        # n_max = 6830
        # loader_tr = dp.MaxPackedBatchLoader(
        #     dataset_tr, n_max, mask=True, batch_size=batch_size, shuffle=False
        # )
        # loader_val = dp.MaxPackedBatchLoader(
        #     dataset_val, n_max, mask=True, batch_size=batch_size, shuffle=False
        # )
        # loader_test = dp.MaxPackedBatchLoader(
        #     dataset_test, n_max, mask=True, batch_size=batch_size, shuffle=False
        # )
        loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, shuffle=False)
        loader_val = BatchLoader(dataset_val, batch_size=batch_size, shuffle=False)
        loader_test = BatchLoader(dataset_test, batch_size=batch_size, shuffle=False)

        self.dataset = [loader_tr, loader_val, loader_test]
        return

    def train_model(self, batch_size, epochs, patience):
        if self.dataset is None:
            self.prepare_train_val_test(batch_size)

        loader_tr, loader_val, loader_test = (
            self.dataset[0],
            self.dataset[1],
            self.dataset[2],
        )
        mtr, btr = self.test_pred(loader_tr, self.mask)
        mval, bval = self.test_pred(loader_val, self.mask)
        mte, bte = self.test_pred(loader_test, self.mask)
        print(
            "Sharpe Ratio Training: {:.4f}, Validation: {:.4f}, Test: {:.4f}".format(
                -1 * mtr, -1 * mval, -1 * mte
            )
        )

    def test_pred(self, loader, mask):
        model_loss = 0
        curr_batch = 0
        batch_loss = []
        loss_fn = CustomSharpe(expand_dims=True)
        loader_count = 0
        for inputs, target in loader:
            loader_count += 1
            if mask:
                nx_measures = inputs[0][:, :, :-1]
            else:
                nx_measures = inputs[0][:, :, 0:]  # get centrality measures from graph
            nodes_ind = 1 / np.mean(nx_measures, axis=-1)
            nodes_ind[nodes_ind == np.inf] = 0
            wgts = np.round(nodes_ind / np.sum(nodes_ind), 6)
            bl = loss_fn(target, wgts)
            bl = bl.numpy()
            batch_loss.append(bl)
            model_loss += bl
            curr_batch += 1

            if curr_batch == loader.steps_per_epoch:
                break
            break
        model_loss /= curr_batch
        return model_loss, batch_loss
