import os
import logging
import time
import glob


import hydra
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from sklearn import preprocessing

from graph_port.data_processing import file_position, pad_list, MaxPackedBatchLoader
import spektral as sp
from graph_port.gnn_models import GN_Model
from graph_port.utils import CustomSharpe


log = logging.getLogger(__name__)
n_max = 6830


class SingDataset(sp.data.Dataset):
    def __init__(
        self, filelist, **kwargs,
    ):
        self.filelist = filelist
        super().__init__(**kwargs)

    def download(self):
        pass

    def read(self):
        # We must return a list of Graph objects
        output = []
        try:
            for fname in self.filelist:
                with np.load(fname, allow_pickle=True) as data:
                    y = data["y"]
                    np_y = np.array(list(map(lambda x: pad_list(x, n_max), y)))
                    normalizer = preprocessing.StandardScaler().fit(data["x_ret"])
                    x = normalizer.transform(data["x_ret"]).T
                    np_y = normalizer.transform(np_y)
                    output.append(
                        sp.data.Graph(
                            x=x,  # data["x_ret"].T[:, :700],
                            # x=data["x"],
                            a=data["a"],
                            e=data["e"],
                            y=np_y,  # , x_tic=data["x_tic"]
                        )
                    )
        except Exception as e:
            print(e.__doc__)
            print(str(e))

        return output


class Trainer:
    def __init__(
        self, model, train_dataset, val_dataset=None, loss_fn=None, optimizer=None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = CustomSharpe(trading_year_days=4) if loss_fn is None else loss_fn
        self.optimizer = (
            tf.keras.optimizers.Adam(lr=1e-4) if optimizer is None else optimizer
        )
        self.best_weights = None

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
            loss_value += sum(self.model.losses)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        val_loss = self.loss_fn(y, val_logits)
        val_loss += sum(self.model.losses)
        return val_loss

    # @profile
    def train(self, batch_size, epochs, patience, datset_test):

        metric_names = [
            "loss",
        ]
        wait = 0
        best = 0
        overall_epoch = 0
        print(
            "Started training with epochs: ",
            epochs,
            "batch_size: ",
            batch_size,
            "patience: ",
            patience,
        )
        for tr, val in zip(self.train_dataset, self.val_dataset):
            overall_length = len(self.train_dataset)
            start_time = time.time()
            print("\nOverall progress {}/{}".format(overall_epoch, overall_length))
            print(tr.numpy())
            print(val.numpy())
            graph_tr = SingDataset(tr.numpy())
            graph_val = SingDataset(val.numpy())

            loader_tr = MaxPackedBatchLoader(
                graph_tr, n_max=n_max, mask=True, batch_size=batch_size, shuffle=False,
            )
            loader_val = MaxPackedBatchLoader(
                graph_val, n_max=n_max, mask=True, batch_size=batch_size, shuffle=False,
            )

            wait = 0
            best = 0

            for epoch in range(epochs):
                print("\nepoch {}/{}".format(epoch + 1, epochs))
                # Iterate over the batches of the dataset.

                pb_i = tf.keras.utils.Progbar(batch_size, stateful_metrics=metric_names)
                start_time = time.time()
                for step, (x_batch_train, y_batch_train) in enumerate(loader_tr):
                    loss_value = self.train_step(x_batch_train, y_batch_train)
                    pb_i.add(loader_tr.steps_per_epoch, values=[("loss", loss_value)])
                    if step + 1 == batch_size:
                        break

                # Run a validation loop at the end of each epoch.
                batch_loss = []
                val_loss = 0
                for val_step, (x_batch_val, y_batch_val) in enumerate(loader_val):
                    bl = self.test_step(x_batch_val, y_batch_val).numpy()
                    batch_loss.append(bl)
                    if val_step + 1 == batch_size:
                        break

                val_loss = np.mean(batch_loss)
                print(" val_loss: %.4f" % (float(val_loss)))
                print("Time taken: %.2fs" % (time.time() - start_time))

                wait += 1
                if val_loss < best:
                    best = val_loss
                    wait = 0
                    self.best_weights = self.model.get_weights()
                if wait >= patience:
                    print("Early stopping: ", best)
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    break

            overall_epoch += 1

            self.test(datset_test)
        print("**Training model finished**")

    def test(self, loader_test):
        test_loss = []
        print("Test data: ")
        # for test_epoch, ba in enumerate(loader):
        #     graph_test = SingDataset(ba.numpy())
        #     loader_test = MaxPackedBatchLoader(
        #         graph_test, n_max=n_max, mask=True, shuffle=False
        #     )
        for step, (x_batch_test, y_batch_test) in enumerate(loader_test):
            bl = self.test_step(x_batch_test, y_batch_test)
            test_loss.append(bl.numpy())
            if step + 1 == loader_test.steps_per_epoch:
                break

        print(test_loss)
        print("Overall: ", np.mean(test_loss))

        return np.mean(test_loss)


# @profile
@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig) -> None:
    path = cfg.app_params.data_path
    batch_size = cfg.model_params.batch_size
    epochs = cfg.model_params.epochs
    patience = cfg.model_params.patience
    trading_year_days = cfg.app_params.trading_year_days
    n_max = cfg.app_params.n_max

    file_list = glob.glob1(path, "*.npz")
    slist = sorted(file_list, key=file_position)
    sorted_file_list = [os.path.join(path, fname) for fname in slist]

    dataset = tf.data.Dataset.from_tensor_slices(sorted_file_list)

    train_size = int(0.85 * len(dataset))
    dataset_tr = dataset.take(train_size - batch_size)
    dataset_val = dataset.skip(batch_size)
    dataset_val = dataset_val.take(train_size - batch_size)

    dataset_test = dataset.skip(train_size)
    dataset_tr = dataset_tr.batch(batch_size)
    dataset_val = dataset_val.batch(batch_size)
    dataset_test = dataset_test.batch(len(dataset_test))

    model = GN_Model(n_max, mask=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = CustomSharpe(trading_year_days=trading_year_days)
    tr = Trainer(model, dataset_tr, dataset_val, loss_fn, optimizer)

    for test_epoch, ba in enumerate(dataset_test):
        graph_test = SingDataset(ba.numpy())
        loader_test = MaxPackedBatchLoader(
            graph_test, n_max=n_max, mask=True, shuffle=False
        )

    tr.train(batch_size, epochs, patience, loader_test)


if __name__ == "__main__":
    main()
