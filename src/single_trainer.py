from locale import normalize
import os
import logging
import traceback
import glob
import hydra
import datetime
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import spektral as sp
from sklearn import preprocessing

from omegaconf import DictConfig
from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

from graph_port.data_processing import file_position
from graph_port.gnn_models import GN_Model
from graph_port.utils import CustomSharpe, SharpeLoss


log = logging.getLogger(__name__)

PARENT_DIR = r"/scratch/krk1g19/prj_3/2023"
print("PARENT_DIR: ", PARENT_DIR)
tf.keras.backend.set_floatx("float32")


class SingDataset(sp.data.Dataset):
    def __init__(
        self,
        filepath,
        start_end=[0, 1],
        x_type=True,
        **kwargs,
    ):
        self.filepath = filepath
        self.x_type = x_type
        self.start_end = np.array(start_end)
        super().__init__(**kwargs)

    def download(self):
        pass

    def read(self):
        # We must return a list of Graph objects
        output = []
        try:
            with np.load(self.filepath, allow_pickle=True) as data:
                y_sl = np.array(data["y"].shape[0] * self.start_end, dtype=int)
                sy = slice(y_sl[0], y_sl[1], None)
                y = data["y"][sy]
                output.append(
                    sp.data.Graph(
                        x=data["x_ret"].T if self.x_type else data["x"],
                        #   x=data['x'],
                        a=data["a"],
                        e=data["e"],
                        # y=np_y,  #
                        x_tic=data["x_tic"],
                        y=y,
                    )
                )
        except Exception as e:
            print(e.__doc__)
            print(str(e))

        return output


class Trainer:
    def __init__(
        self,
        model,
        loader_tr,
        loader_val,
        loader_test,
        loss_fn,
        optimizer,
        metric_fn=None,
        log_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        graph_id=None,
    ):
        self.model = model
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.best_weights = None
        self.graph_id = str(graph_id)

        train_log_dir = os.path.join(
            PARENT_DIR,
            "logs/gradient_tape/" + log_time + "/" + self.graph_id + "/train",
        )
        test_log_dir = os.path.join(
            PARENT_DIR, "logs/gradient_tape/" + log_time + "/" + self.graph_id + "/val"
        )

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
            loss_value += sum(self.model.losses)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

    # @tf.function(experimental_relax_shapes=True)
    def val_step(self, x, y):
        val_logits = self.model(x, training=False)
        val_loss = self.loss_fn(y, val_logits)
        val_loss += sum(self.model.losses)
        return val_loss

    # @tf.function(experimental_relax_shapes=True)
    def test_step(self, x, y):
        test_logits = self.model(x, training=False)
        test_loss = self.loss_fn(y, test_logits)
        test_loss += sum(self.model.losses)
        return test_loss, test_logits

    def train(self, epochs, patience):
        wait = 0
        best_val = 0
        best_tr = 0
        for epoch in range(epochs):
            print("\nepoch {}/{}".format(epoch + 1, epochs))
            # Iterate over the batches of the dataset.

            # pb_i = tf.keras.utils.Progbar(batch_size, stateful_metrics=metric_names)
            train_loss = 0
            val_loss = 0
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(self.loader_tr):
                train_loss = self.train_step(x_batch_train, y_batch_train).numpy()
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss, step=epoch)
                break

            for step, (x_batch_val, y_batch_val) in enumerate(self.loader_val):
                val_loss = self.val_step(x_batch_val, y_batch_val).numpy()
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("loss", val_loss, step=epoch)
                break

            print(
                "Time: %.2f train_loss: %.4f val_loss: %.4f"
                % (float(time.time() - start_time), float(train_loss), float(val_loss))
            )

            if epoch <= 2:  #  wait 2 epochs and start patience
                best_val = val_loss
                best_tr = train_loss
            else:
                wait += 1
                if val_loss < best_val:
                    best_val = val_loss
                    best_tr = train_loss
                    wait = 0
                    self.best_weights = self.model.get_weights()
                if wait >= patience:
                    print("Early stopping: ", best_val)
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    break

        test_loss = 0
        for test_step, (x_batch_test, y_batch_test) in enumerate(self.loader_test):
            test_loss, test_logits = self.test_step(x_batch_test, y_batch_test)
            test_loss = test_loss.numpy()
            break
        print(
            "train_loss: %.4f val_loss: %.4f test_loss: %.4f"
            % (best_tr, best_val, test_loss)
        )
        return (
            {"train": -1 * best_tr, "val": -1 * best_val, "test": -1 * test_loss},
            test_logits,
        )

    def infer(self):
        train_sharpe = 0
        val_sharpe = 0
        test_sharpe = 0
        for (x_batch_train, y_batch_train) in self.loader_tr:
            _, train_logits = self.test_step(x_batch_train, y_batch_train)
            train_sharpe = self.metric_fn(y_batch_train, train_logits).numpy()
            break

        for (x_batch_val, y_batch_val) in self.loader_val:
            _, val_logits = self.test_step(x_batch_val, y_batch_val)
            val_sharpe = self.metric_fn(y_batch_val, val_logits).numpy()
            break

        for (x_batch_test, y_batch_test) in self.loader_test:
            _, test_logits = self.test_step(x_batch_test, y_batch_test)
            test_sharpe = self.metric_fn(y_batch_test, test_logits).numpy()
            break

        print(
            "train_sharpe: %.4f val_sharpe: %.4f test_sharpe: %.4f"
            % (train_sharpe, val_sharpe, test_sharpe)
        )
        return {
            "train": -1 * train_sharpe,
            "val": -1 * val_sharpe,
            "test": -1 * test_sharpe,
        }


class EqualTrainer:
    def __init__(self, loader_tr, loader_val, loader_test, loss_fn=None):
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.loss_fn = (
            CustomSharpe(trading_year_days=4.0, loss_mode="test")
            if loss_fn is None
            else loss_fn
        )

    def train_eq_step(self, loader):
        for test_step, (x, y) in enumerate(loader):
            wgt_arr = np.ones((1, y.shape[1]))
            wgt_arr = np.round(wgt_arr / np.sum(wgt_arr), decimals=6).T
            loss = self.loss_fn(y, wgt_arr)
            break
        return loss.numpy(), wgt_arr

    def train(self):
        btr, _ = self.train_eq_step(self.loader_tr)
        bval, _ = self.train_eq_step(self.loader_val)
        btest, wgt_arr = self.train_eq_step(self.loader_test)
        print("train_loss: %.4f val_loss: %.4f test_loss: %.4f" % (btr, bval, btest))
        return {"train": -1 * btr, "val": -1 * bval, "test": -1 * btest}, wgt_arr


class NetworkTrainer:
    def __init__(self, loader_tr, loader_val, loader_test, loss_fn=None):
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.loss_fn = (
            CustomSharpe(trading_year_days=4.0, loss_mode="test")
            if loss_fn is None
            else loss_fn
        )

    def train_eq_step(self, loader):
        for test_step, (x, y) in enumerate(loader):
            nx_measures = x[0][:, 0:]
            # nodes_ind = np.mean(nx_measures, axis=-1)
            nodes_ind = 1 / np.mean(nx_measures, axis=-1)
            nodes_ind[nodes_ind == np.inf] = 0
            wgts = np.round(nodes_ind / np.sum(nodes_ind), 6).T
            wgts = np.expand_dims(wgts, -1)
            loss = self.loss_fn(y, wgts)
            break
        return loss.numpy(), wgts

    def train(self):
        btr, _ = self.train_eq_step(self.loader_tr)
        bval, _ = self.train_eq_step(self.loader_val)
        btest, wgt_arr = self.train_eq_step(self.loader_test)
        print("train_loss: %.4f val_loss: %.4f test_loss: %.4f" % (btr, bval, btest))
        return {"train": -1 * btr, "val": -1 * bval, "test": -1 * btest}, wgt_arr


class EffPortTrainer:
    def __init__(self, loader_tr, loader_val, loader_test, loss_fn=None):
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.loss_fn = (
            CustomSharpe(trading_year_days=4.0, loss_mode="test")
            if loss_fn is None
            else loss_fn
        )
        self.wgts = None

    def train_eq_step(self, loader):
        for test_step, (x, y) in enumerate(loader):
            try:
                if self.wgts is None:
                    inputs = pd.DataFrame(x[0].T)

                    print("inputs: ", inputs.shape, "y: ", y.shape)
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
                    vals = np.zeros((1, len(weights_ef)))
                    for i, key in enumerate(weights_ef.keys()):
                        vals[0, i] = weights_ef[key]
                    self.wgts = vals.T

                loss = self.loss_fn(y, self.wgts)
                return loss.numpy(), self.wgts

            except Exception as e:
                print(e.__doc__)
                print(str(e))
                print(traceback.print_exc())
                return 0, np.zeros((y.shape[1], 1))

    def train(self):
        btr, _ = self.train_eq_step(self.loader_tr)
        bval, _ = self.train_eq_step(self.loader_val)
        btest, wgt_arr = self.train_eq_step(self.loader_test)
        print("train_loss: %.4f val_loss: %.4f test_loss: %.4f" % (btr, bval, btest))
        return {"train": -1 * btr, "val": -1 * bval, "test": -1 * btest}, wgt_arr


def get_dataset(dirpath):
    file_list = glob.glob1(dirpath, "*.npz")
    slist = sorted(file_list, key=file_position)
    sorted_file_list = [os.path.join(dirpath, fname) for fname in slist]
    dataset = tf.data.Dataset.from_tensor_slices(sorted_file_list)
    dataset = dataset.batch(1)

    return dataset


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig) -> None:

    dirpath = cfg.app_params.data_path
    epochs = cfg.model_params.epochs
    patience = cfg.model_params.patience
    trading_year_days = cfg.app_params.trading_year_days
    split_va = cfg.model_params.validation_split
    split_te = cfg.model_params.test_split
    x_type = cfg.app_params.x_type
    graph_res = pd.DataFrame(columns=["train", "val", "test"])
    eq_res = pd.DataFrame(columns=["train", "val", "test"])
    nw_res = pd.DataFrame(columns=["train", "val", "test"])
    ep_res = pd.DataFrame(columns=["train", "val", "test"])

    dataset = get_dataset(dirpath)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    alloc_log_dir = os.path.join(
        PARENT_DIR, "logs/gradient_tape/" + current_time + "/allocation"
    )
    alloc_summary_writer = tf.summary.create_file_writer(alloc_log_dir)

    for i, data in enumerate(dataset):

        fname = data.numpy()[0]
        print("tr: ", fname)
        graph_id = file_position(str(fname, "unicode_escape"))
        print("graph_id: ", graph_id)
        graph_tr = SingDataset(fname, [0, split_va], x_type)
        x_tic = graph_tr.graphs[0].x_tic
        # normalizer = preprocessing.StandardScaler().fit(graph_tr.graphs[0]["y"])
        # graph_tr.graphs[0]["y"] = normalizer.transform(graph_tr.graphs[0]["y"])
        graph_val = SingDataset(fname, [split_va, split_te], x_type)
        # graph_val.graphs[0]["y"] = normalizer.transform(graph_val.graphs[0]["y"])
        graph_test = SingDataset(fname, [split_te, 1], x_type)
        # graph_test.graphs[0]["y"] = normalizer.transform(graph_test.graphs[0]["y"])

        loader_tr = sp.data.SingleLoader(graph_tr)
        loader_val = sp.data.SingleLoader(graph_val)
        loader_test = sp.data.SingleLoader(graph_test)

        model = GN_Model(mask=False)
        loss_fn = SharpeLoss(
            trading_year_days=trading_year_days
        )  # CustomSharpe(trading_year_days=trading_year_days)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        metric_fn = CustomSharpe(trading_year_days=trading_year_days)
        model.compile(optimizer=optimizer, loss=loss_fn)
        # if i == 0:
        model.model_summary(
            (
                graph_tr.graphs[0].x.shape,
                graph_tr.graphs[0].a.shape,
                graph_tr.graphs[0].e.shape,
            )
        )
        model = GN_Model(mask=False)
        model.compile(optimizer=optimizer, loss=loss_fn)

        tr = Trainer(
            model,
            loader_tr,
            loader_val,
            loader_test,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metric_fn=metric_fn,
            graph_id=graph_id,
            log_time=current_time,
        )
        res, wgts = tr.train(epochs, patience)
        res = tr.infer()

        graph_res = graph_res.append(res, ignore_index=True)

        # et = EqualTrainer(loader_tr, loader_val, loader_test, loss_fn)
        # re, eq_wgts = et.train()
        # eq_res = eq_res.append(re, ignore_index=True)

        # nt = NetworkTrainer(loader_tr, loader_val, loader_test, loss_fn)
        # rne, nw_wgts = nt.train()
        # nw_res = nw_res.append(rne, ignore_index=True)

        # ep = EffPortTrainer(loader_tr, loader_val, loader_test, loss_fn)
        # rep, ep_wgts = ep.train()
        # ep_res = ep_res.append(rep, ignore_index=True)

        graph_res.to_csv(os.path.join(PARENT_DIR, "results_pract/graph_results.csv"))
        # eq_res.to_csv(os.path.join(PARENT_DIR, "results/equal_results.csv"))
        # nw_res.to_csv(os.path.join(PARENT_DIR, "results/nw_results.csv"))
        # ep_res.to_csv(os.path.join(PARENT_DIR, "results/ep_results.csv"))

        # print(wgts.shape, eq_wgts.shape, x_tic.shape)
        wgt_column = "graph_wgts"
        wgtsdf = pd.DataFrame(
            {wgt_column: wgts.numpy().squeeze()},
            index=x_tic,
        )
        # wgtsdf = pd.DataFrame(
        #     {"graph_wgts": wgts.numpy().squeeze(), "eq_wgts": eq_wgts.squeeze()},
        #     index=x_tic,
        # )
        # wgtsdf = pd.DataFrame({"ep_wgts": ep_wgts.squeeze()}, index=x_tic,)
        wgtsdf.to_csv(
            os.path.join(
                PARENT_DIR, "results_pract/wgts/wgts_" + str(graph_id) + "_.csv"
            ),
            index_label="tic",
        )
        zero_wgts = (wgtsdf[wgt_column] == 0).sum() / wgtsdf.shape[0]
        max_wgt = wgtsdf[wgt_column].max()
        with alloc_summary_writer.as_default():
            tf.summary.scalar("zero_wgts_alloc", zero_wgts, step=graph_id)
            tf.summary.scalar("maximum_wgt", max_wgt, step=graph_id)
        print(
            "Zero Allocations: {:0.2%} Max Weight: {:0.4%}".format(zero_wgts, max_wgt)
        )


if __name__ == "__main__":
    main()
