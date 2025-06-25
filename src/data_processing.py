import numpy as np
import pandas as pd
import networkx as nx
import spektral as sp
import tensorflow as tf
import graph_port.utils as ut
import os, glob, re
import progressbar as pb
import shutil

import logging

log = logging.getLogger("__main__")


def get_prep_data(path, pivot=False):
    if pivot:
        ind_df = pd.read_excel(path, sheet_name="Sheet1")
        ind_df["Date"] = pd.to_datetime(ind_df["Date"], utc=True).dt.date
        ind_df.set_index("Date", inplace=True)
        ind_df.index = pd.to_datetime(ind_df.index)

        px_df = ind_df.pivot_table(
            index=ind_df.index, columns=["Symbol"], values=["close_price"]
        )
        px_df.columns = px_df.columns.levels[1]
        cols = [
            ".BVLG",
            ".FTMIB",
            ".GSPTSE",
            ".OMXC20",
            ".OMXHPI",
            ".OMXSPI",
            ".OSEAX",
            ".SMSI",
        ]
        px_df.drop(columns=cols, inplace=True, errors="ignore")
    else:
        px_df = pd.read_excel(path, sheet_name="Sheet1")
        px_df["Date"] = pd.to_datetime(px_df["Date"], utc=True).dt.date
        px_df.set_index("Date", inplace=True)
        px_df.index = pd.to_datetime(px_df.index)
        px_df.fillna(method="ffill", inplace=True)

    # lret = lambda x: np.log(x).diff()
    # ret_df= px_df.apply(lret)
    ret_df = px_df.pct_change(axis=0)
    drop_index = ret_df.index[ret_df.isna().all(axis=1)]
    px_df.drop(index=drop_index, inplace=True)
    ret_df.drop(index=drop_index, inplace=True)
    ret_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ret_df.fillna(method="ffill", inplace=True)
    ret_df.fillna(0, inplace=True)

    # vol_df=ind_df.pivot_table(index=ind_df.index,columns=['Symbol'],values=['medrv'])
    # vol_df.columns=vol_df.columns.levels[1]
    # vol_df.fillna(method='ffill',inplace=True)
    # vol_df.fillna(0,inplace=True)

    vol_df = ret_df.rolling(window=30).std() * np.sqrt(252)
    drop_index = vol_df.index[vol_df.isna().all(axis=1)]
    px_df.drop(index=drop_index, inplace=True)
    ret_df.drop(index=drop_index, inplace=True)
    vol_df.drop(index=drop_index, inplace=True)

    vol_df.fillna(method="ffill", inplace=True)
    vol_df.fillna(0, inplace=True)

    print("px_df: ", px_df.shape, "ret_df: ", ret_df.shape, "vol_df: ", vol_df.shape)
    return px_df, ret_df, vol_df


def prepare_rolling_series(
    x_in, y_in, x_win_size=4, forecast_ahead=0, y_win_size=1, skiprows=1
):
    if x_in.shape[0] != y_in.shape[0]:
        assert "Shapes should be equal"

    X = x_in[:-forecast_ahead] if forecast_ahead > 0 else x_in
    if y_win_size == 1 or y_win_size == 0:
        Y1 = (
            y_in[x_win_size + forecast_ahead :]
            if y_in.shape[0] > x_win_size + forecast_ahead
            else y_in[-1:]
        )
    else:
        X = X[: -y_win_size + 1]
        Y1 = y_in[x_win_size + forecast_ahead :]
        no_of_rows = Y1.shape[0]
        if no_of_rows <= y_win_size:
            yroll = np.zeros([1, y_win_size, Y1.shape[1]], dtype=np.float32)
            yroll[0][y_win_size - no_of_rows :] = Y1
            samplesize = 1
        else:
            samplesize = 0
            yroll = np.zeros([no_of_rows, y_win_size, Y1.shape[1]])
            for i in range(0, no_of_rows):
                if i + y_win_size <= no_of_rows:
                    yroll[i] = Y1[i : i + y_win_size]
                    samplesize += 1

        Y1 = yroll[:samplesize]

    no_of_rows = X.shape[0] - 1

    if no_of_rows <= x_win_size:
        xroll = np.zeros([1, x_win_size, X.shape[1]])
        xroll[0][x_win_size - no_of_rows :] = X
        samplesize = 1
    else:
        samplesize = 0
        xroll = np.zeros([no_of_rows, x_win_size, X.shape[1]], dtype=np.float32)
        for i in range(0, no_of_rows):
            if i + x_win_size <= no_of_rows:
                xroll[i] = X[i : i + x_win_size]
                samplesize += 1

    xdata = xroll[:samplesize]

    if skiprows > 1:
        xdata = xdata[::skiprows]
        Y1 = Y1[::skiprows]
        print("X: ", xdata.shape, "y: ", Y1.shape)

    return xdata, Y1


def get_spektral_graph_from_matrix(
    tabmat, node_f, graph_type, node_labels=None, y=None
):
    G = nx.from_numpy_matrix(np.matrix(tabmat), create_using=nx.DiGraph)
    if node_labels is not None:
        mapping = dict(zip(range(node_labels.shape[0]), node_labels))
        G = nx.relabel_nodes(G, mapping)

    G.remove_edges_from(nx.selfloop_edges(G))
    # _filter_nx_graph(G)
    wgt_adjmat = nx.to_numpy_matrix(G)

    adjmat = nx.to_numpy_array(G, weight=None)

    central_measure = np.zeros((len(node_labels), 3))
    # ec=nx.algorithms.centrality.eigenvector_centrality(G)    #append centraility measures as features of nodes
    # cbc = nx.communicability_betweenness_centrality(G)
    dc = nx.algorithms.centrality.degree_centrality(G)
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)

    for i, lab in enumerate(node_labels):
        central_measure[i] = [cc[lab], bc[lab], dc[lab]]  # ,cbc[lab]]

    if graph_type == "TMFG":
        node_wgts = central_measure
    else:
        node_wgts = np.sum(wgt_adjmat, axis=0).T
        node_attr = dict(zip(G.nodes, node_wgts.squeeze().tolist()[0]))
        nx.set_node_attributes(G, node_attr, "To")
        node_wgts = np.append(node_wgts, central_measure, axis=1)

    # zip_iterator = zip(node_labels, node_f)   #Append the time series
    # attr_dict=dict(zip_iterator)
    # nx.set_node_attributes(G,attr_dict,'Returns')
    # node_wgts = np.append(node_wgts,node_f,axis=1)
    return (
        G,
        sp.data.graph.Graph(x=node_wgts, a=adjmat, e=wgt_adjmat[:, :, np.newaxis], y=y),
    )


# internal function to
#  reduce graph edges , TODO: add planarity removal later on
def _filter_nx_graph(G):
    edge_weights = nx.get_edge_attributes(G, "weight")
    med_weight = np.median(list(edge_weights.values()))

    G.remove_edges_from((e for e, w in edge_weights.items() if w < med_weight))


class VolDataset(sp.data.Dataset):
    def __init__(
        self,
        path,
        n_max=None,
        input_data=None,
        node_features=None,
        node_labels=None,
        graph_type="CONNECTED",
        start_range=0,
        **kwargs,
    ):
        sp.data.Dataset.path = path
        self.x = None
        self.y = None
        self.node_f = None
        self.graph_size = 0
        self.refresh_data = False
        self.nodel_labels = None
        self.graph_type = None

        if input_data is not None:
            print("Build graph: ", graph_type)
            self.x = input_data[0]
            self.y = input_data[1]
            self.node_f = node_features
            self.graph_size = input_data[0].shape[0]
            self.refresh_data = True
            self.node_labels = node_labels
            self.graph_type = graph_type

        self.start_range = start_range

        self.n_max = n_max

        if self.refresh_data and os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)

        super().__init__(**kwargs)

    def download(self):
        # Create the directory
        os.mkdir(self.path)
        pos = None
        for i in pb.progressbar(range(self.graph_size)):
            try:
                rinput = self.x[i].squeeze()
                if self.graph_type == "TMFG":
                    tabmat = ut.compute_tfmg_adj(rinput)
                else:
                    tabmat = ut.compute_connectedness(rinput)

                G, sg = get_spektral_graph_from_matrix(
                    tabmat,
                    self.node_f[i].T,
                    self.graph_type,
                    self.node_labels,
                    self.y[i],
                )
                filename = os.path.join(self.path, f"graph_{i+self.start_range}")
                graphfilename = os.path.join(
                    self.path, f"graphviz_{i+self.start_range}"
                )
                np.savez(
                    filename,
                    x=sg.x,
                    a=sg.a,
                    e=sg.e,
                    y=sg.y,
                    x_ret=rinput,
                    x_tic=self.node_labels,
                )

                # if pos is None:
                #     pos = ut.plt_corr_nx(G,'Graph_'+str(i),graphfilename,self.graph_type)
                # else:
                #     ut.plt_corr_nx(G,'Graph_'+str(i),graphfilename,self.graph_type,pos)#

            except Exception as e:
                print(e.__doc__)
                print(str(e))
                continue

    def read(self):
        # We must return a list of Graph objects
        output = []
        file_list = glob.glob1(self.path, "*.npz")
        slist = sorted(file_list, key=file_position)

        for fname in pb.progressbar(slist):
            try:
                with np.load(os.path.join(self.path, fname), allow_pickle=True) as data:
                    y = data["y"]
                    if self.n_max is not None:
                        np_y = np.array(list(map(lambda x: pad_list(x, 6830), y)))
                        output.append(
                            sp.data.Graph(
                                x=data["x_ret"].T[:, :700],
                                a=data["a"],
                                e=data["e"],
                                y=np_y,  # , x_tic=data["x_tic"]
                            )
                        )
                    else:
                        output.append(
                            sp.data.Graph(
                                x=data["x"],
                                a=data["a"],
                                e=data["e"],
                                y=y,
                            )
                        )

            except Exception as e:
                print(e.__doc__)
                print(str(e))
                continue
        return output


def pad_list(yi, length):
    yi = np.pad(yi, (0, length - yi.shape[0]), constant_values=0)
    return yi


class SeriesDataset:
    def __init__(self, path, n_max=None, **kwargs):
        self.path = path
        self.n_max = n_max
        self.inp_data = self.read()

    def read(self):
        X = []
        y = []
        x_tic = []

        file_list = glob.glob1(self.path, "*.npz")
        slist = sorted(file_list, key=file_position)

        for fname in pb.progressbar(slist):
            try:
                with np.load(os.path.join(self.path, fname), allow_pickle=True) as data:

                    if self.n_max is not None:
                        x_tic.append(data["x_tic"])
                        X.append(data["x_ret"])
                    else:
                        X.append(data["x"])

                    y.append(data["y"])
            except Exception as e:
                print(e.__doc__)
                print(str(e))
                continue

        return (
            X,
            y,
        )  # (tf.data.Dataset.from_tensor_slices((X, y)),X[0].shape,y[0].shape)


def get_lstm_data(data_path, data_cfg):
    px_mkt, ret_mkt, vol_mkt = get_prep_data(data_path, data_cfg.pivot_type)
    X, y = prepare_rolling_series(
        ret_mkt.values,
        ret_mkt.values,
        data_cfg.lookback,
        data_cfg.forecast_ahead,
        data_cfg.forecast_size,
        data_cfg.skip_rows,
    )
    return (X, y)


def get_ep_data(data_path, data_cfg):
    px_mkt, ret_mkt, vol_mkt = get_prep_data(data_path, data_cfg.pivot_type)
    X, y = prepare_rolling_series(
        ret_mkt.values,
        ret_mkt.values,
        data_cfg.lookback,
        data_cfg.forecast_ahead,
        data_cfg.forecast_size,
        data_cfg.skip_rows,
    )
    return (X, y)


def file_position(x, pattern="_(.*?).npz"):
    # pattern =
    xname = os.path.basename(x)
    filidx = int(re.search(pattern, xname).group(1))
    return filidx


def file_key(x):
    pattern = "_(.*?)_.xlsx"
    filidx = int(re.search(pattern, x).group(1))
    return filidx


def get_data_from_path(cfg):

    data = None
    if cfg.app_params.model == "GNN" or cfg.app_params.model == "NWX":
        data = VolDataset(cfg.app_params.data_path, cfg.app_params.n_max)
    elif cfg.app_params.model == "LSTM":
        # data = get_lstm_data(cfg.app_params.data_path,cfg.data_params)
        data = SeriesDataset(cfg.app_params, cfg.app_params.n_max)
    elif (
        cfg.app_params.model == "EFFICIENT_PORTFOLIO"
        or cfg.app_params.model == "EQUAL_PORTFOLIO"
    ):
        # data = get_ep_data(cfg.app_params.data_path,cfg.data_params)
        data = SeriesDataset(cfg.app_params.data_path, cfg.app_params.n_max)

    return data


def build_graphdataset(file_path, graph_dir, data_cfg):
    px_mkt, ret_mkt, vol_mkt = get_prep_data(file_path, data_cfg.pivot_type)
    print(
        "px: ",
        px_mkt.shape,
        "ret: ",
        ret_mkt.shape,
        "vol: ",
        vol_mkt.shape,
        "graph type: ",
        data_cfg.graph_type,
    )

    ret_X, ret_y = prepare_rolling_series(
        ret_mkt.values,
        ret_mkt.values,
        data_cfg.lookback,
        data_cfg.forecast_ahead,
        data_cfg.forecast_size,
        data_cfg.skip_rows,
    )
    vol_X, vol_y = prepare_rolling_series(
        vol_mkt.values,
        ret_mkt.values,
        data_cfg.lookback,
        data_cfg.forecast_ahead,
        data_cfg.forecast_size,
        data_cfg.skip_rows,
    )  # y stil return values
    return VolDataset(
        graph_dir, (vol_X, vol_y), ret_X, vol_mkt.columns, data_cfg.graph_type
    )


def build_graphdataset_fromdir(cfg):

    source_dir = cfg.app_params.data_path
    graph_dir = cfg.app_params.graph_dir
    corr_dir = cfg.app_params.corr_dir
    data_cfg = cfg.data_params
    job_id = cfg.job_id
    r_port = cfg.Rserve_port

    file_list = glob.glob1(source_dir, "*.xlsx")
    slist = sorted(file_list, key=file_key)
    start_f = (job_id - 1) * 10
    end_f = job_id * 10
    print("Process files between ", start_f, "and ", end_f)

    for fname in pb.progressbar(slist):
        try:
            i = file_key(fname)

            if (
                i < start_f
                or i >= end_f
                or is_processed(i, data_cfg.processed_file_path)
            ):  # outside job id array don't process or already processed
                continue

            file_path = os.path.join(source_dir, fname)
            corr_filename = os.path.join(corr_dir, f"corr_{i}.npz")

            px_mkt, ret_mkt, vol_mkt = get_prep_data(file_path, data_cfg.pivot_type)
            X = ret_mkt[: -data_cfg.forecast_size].values
            y = ret_mkt[-data_cfg.forecast_size :].values
            rinput = vol_mkt[: -data_cfg.forecast_size].values
            tics = vol_mkt.columns
            ind = vol_mkt.index

            if data_cfg.build_corr:
                ut.distance_corr(rinput, corr_filename)
                continue

            print("compute tmfg")
            if data_cfg.graph_type == "TMFG":
                tabmat = ut.compute_tfmg(corr_filename, r_port)
            else:
                tabmat = ut.compute_connectedness(rinput)
            G, sg = get_spektral_graph_from_matrix(
                tabmat, X.T, data_cfg.graph_type, tics, y
            )
            filename = os.path.join(graph_dir, f"graph_{i}")
            # graphfilename = os.path.join(graph_dir, f'graphviz_{i}')
            np.savez(
                filename, x=sg.x, a=sg.a, e=sg.e, y=sg.y, x_ret=X, x_tic=tics, x_ind=ind
            )
        except Exception as e:
            print(e.__doc__)
            print(str(e))
            continue


def is_processed(file_id, file_path):
    df = pd.read_excel(file_path)
    if file_id in df.values:
        return True
    else:
        return False


def build_graphdataset_fromfile(cfg):

    source_fle = cfg.app_params.data_path
    graph_file = cfg.app_params.graph_dir
    corr_file = cfg.app_params.corr_dir
    data_cfg = cfg.data_params
    job_id = cfg.job_id

    file_path = source_file
    px_mkt, ret_mkt, vol_mkt = get_prep_data(file_path, data_cfg.pivot_type)
    X = ret_mkt[: -data_cfg.forecast_size].values
    y = ret_mkt[-data_cfg.forecast_size :].values
    rinput = vol_mkt[: -data_cfg.forecast_size].values
    tics = vol_mkt.columns
    ind = vol_mkt.index

    if data_cfg.graph_type == "TMFG":
        tabmat = ut.compute_tfmg_adj(rinput, corr_file, 0, data_cfg.print_corr)
    else:
        tabmat = ut.compute_connectedness(rinput)

    print("dtype", tabmat.dtype)
    tabmat = tabmat.astype("float32")
    G, sg = get_spektral_graph_from_matrix(tabmat, X.T, data_cfg.graph_type, tics, y)
    print("Creating npz file: ", filename)
    np.savez(graph_file, x=sg.x, a=sg.a, e=sg.e, y=sg.y, x_ret=X, x_tic=tics, x_ind=ind)


class MaxPackedBatchLoader(sp.data.Loader):
    """
    A `BatchLoader` that zero-pads the graphs before iterating over the dataset.
    This means that `n_max` is computed over the whole dataset and not just
    a single batch.
    While using more memory than `BatchLoader`, this loader should reduce the
    computational overhead of padding each batch independently.
    Use this loader if:
    - memory usage isn't an issue and you want to produce the batches as fast
    as possible;
    - the graphs in the dataset have similar sizes and there are no outliers in
    the dataset (i.e., anomalous graphs with many more nodes than the dataset
    average).
    **Arguments**
    - `dataset`: a graph Dataset;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.
    **Output**
    For each batch, returns a tuple `(inputs, labels)`.
    `inputs` is a tuple containing:
    - `x`: node attributes of shape `[batch, n_max, n_node_features]`;
    - `a`: adjacency matrices of shape `[batch, n_max, n_max]`;
    - `e`: edge attributes of shape `[batch, n_max, n_max, n_edge_features]`.
    `labels` have shape `[batch, ..., n_labels]`.
    """

    def __init__(
        self, dataset, n_max=None, mask=False, batch_size=1, epochs=None, shuffle=True
    ):
        self.n_max = n_max
        self.mask = mask
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

        # Drop the Dataset container and work on packed tensors directly
        packed = self.pack(self.dataset)

        y = packed.pop("y_list", None)
        if y is not None:
            y = np.array(y)

        self.signature = dataset.signature
        self.dataset = ut.to_max_batch(self.n_max, **packed, mask=self.mask)
        if y is not None:
            self.dataset += (y,)

        # Re-instantiate generator after packing dataset
        self._generator = self.generator()

    def collate(self, batch):
        if len(batch) == 2:
            # If there is only one input, i.e., batch = [x, y], we unpack it
            # like this because Keras does not support input lists with only
            # one tensor.
            return batch[0], batch[1]
        else:
            return batch[:-1], batch[-1]

    def tf_signature(self):
        """
        Adjacency matrix has shape [batch, n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_nodes, n_nodes, n_edge_features]
        Targets have shape [batch, ..., n_labels]
        """
        signature = self.signature
        for k in signature:
            signature[k]["shape"] = prepend_none(signature[k]["shape"])
        if "x" in signature:
            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )
        if "a" in signature:
            # Adjacency matrix in batch mode is dense
            signature["a"]["spec"] = tf.TensorSpec
        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature["e"]["shape"] = prepend_none(signature["e"]["shape"])

        return to_tf_signature(signature)

    @property
    def steps_per_epoch(self):
        if len(self.dataset) > 0:
            return int(np.ceil(len(self.dataset[0]) / self.batch_size))
