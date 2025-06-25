import numpy as np
import glob, os
import tensorflow as tf

from graph_port.data_processing import file_position, pad_list, MaxPackedBatchLoader
import spektral as sp
from graph_port.gnn_models import GN_Model
from graph_port.utils import CustomSharpe

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
                    output.append(
                        sp.data.Graph(
                            x=data["x_ret"].T[:, :700],
                            a=data["a"],
                            e=data["e"],
                            y=np_y,  # , x_tic=data["x_tic"]
                        )
                    )
        except Exception as e:
            print(e.__doc__)
            print(str(e))

        return output


def main():

    path = r"/scratch/krk1g19/Datasets/graph_3y3m"

    file_list = glob.glob1(path, "*.npz")
    slist = sorted(file_list, key=file_position)
    sorted_file_list = [os.path.join(path, fname) for fname in slist]

    dataset = tf.data.Dataset.from_tensor_slices(sorted_file_list)

    batch_size = 4
    train_size = int(0.85 * len(dataset))
    dataset_tr = dataset.take(train_size - batch_size)
    dataset_val = dataset.skip(batch_size)
    dataset_val = dataset_val.take(train_size - batch_size)

    dataset_test = dataset.skip(train_size)
    dataset_tr = dataset_tr.batch(batch_size)
    dataset_val = dataset_val.batch(batch_size)
    dataset_test = dataset_test.batch(1)

    model = GN_Model(n_max, mask=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = CustomSharpe(trading_year_days=4.0)
    model.compile(optimizer=optimizer, loss=loss_fn)

    epochs = 150
    patience = 10
    overall_epoch = 0
    for tr, val in zip(dataset_tr, dataset_val):
        print("***start of *** : ", overall_epoch, " dataset")
        graph_tr = SingDataset(tr.numpy())
        graph_val = SingDataset(val.numpy())
        loader_tr = MaxPackedBatchLoader(
            graph_tr, n_max=n_max, mask=True, batch_size=1, epochs=epochs, shuffle=False
        )
        loader_val = MaxPackedBatchLoader(
            graph_val, n_max=n_max, mask=True, batch_size=1, shuffle=False,
        )

        model.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            epochs=epochs,
            validation_data=loader_val.load(),
            validation_steps=loader_val.steps_per_epoch,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=patience, restore_best_weights=True
                )
            ],
        )

        overall_epoch += 1
        test_loss = []
        print("Test data: ")
        for test_epoch, ba in enumerate(dataset_test):
            graph_test = SingDataset(ba.numpy())
            loader_test = MaxPackedBatchLoader(
                graph_test, n_max=n_max, mask=True, shuffle=False
            )

            test_loss.append(test_pred(model, loader_test))
        print(test_loss)

        print("Overall: ", np.mean(test_loss))


def test_pred(model, loader):
    model_loss = 0
    loss_fn = CustomSharpe(trading_year_days=4.0)
    for batch in loader:
        inputs, target = batch
        # print(inputs[0].shape,inputs[1].shape,target.shape)
        predictions = model(inputs, training=False)
        bl = loss_fn(target, predictions)
        model_loss = bl.numpy()
        break

    return model_loss


if __name__ == "__main__":
    main()
