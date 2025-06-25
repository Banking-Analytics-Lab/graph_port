import numpy as np
import pandas as pd
import tensorflow as tf

# import quantstats as qs
import pyRserve

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import dcor
from progressbar import progressbar
from spektral.utils.misc import pad_jagged_array

# qs.extend_pandas()

# metric_list=[f for f in dir(qs.stats) if f[0] != '_']


def calculate_portfolio_returns(wgt, test_ret):
    wgtsdf = pd.DataFrame(index=test_ret.columns, columns=["Weight"])
    wgtsdf["Weight"] = pd.DataFrame.from_dict(wgt, orient="index")
    wgtsdf.fillna(0, inplace=True)

    drtrn = test_ret.dot(wgtsdf)
    drtrn.rename(columns={"Weight": "Return"}, inplace=True)

    return drtrn


# def measure_performance(returns,rf=0,met='sharpe'):
#    return getattr(qs.stats,met)(returns,rf)


def compute_connectedness(vold):
    conn = pyRserve.connect()
    # print(conn)
    try:
        if isinstance(vold, pd.DataFrame):
            conn.r.z_in_vol = vold.values
        else:
            conn.r.z_in_vol = vold

        test_r_script = r"""
                
            library(frequencyConnectedness)
            library(zoo)
            library(BigVAR)
            
        
            big_var_est <- function(data) {
            Model1 = constructModel(as.matrix(data), p = 2, struct = "Basic", gran = c(50, 50), VARX = list(), verbose = F)
            Model1Results = cv.BigVAR(Model1)
            }

            oo <- big_var_est(log(z_in_vol[apply(z_in_vol>0, 1, all),]))

            sp <- spilloverDY12(oo, n.ahead = 20, no.corr = F)
    """

        result = conn.eval(test_r_script)
        tabmat = np.array(result["tables"])
        tabmat = tabmat.squeeze()

        if ~conn.isClosed:
            conn.close()
    except Exception as e:
        conn.close()
        print(e.__doc__)
        print(str(e))
        return None

    return np.matrix(tabmat)


def compute_tfmg_adj(vold, corr_filepath, file_id, print_corr=False):
    conn = pyRserve.connect()
    # print(conn)

    try:
        if isinstance(vold, pd.DataFrame):
            conn.r.rdf = distance_corr(vold.values)
        else:
            conn.r.rdf = distance_corr(vold)

        test_r_script = r"""
            options(warn = -1)
            suppressPackageStartupMessages(library(NetworkToolbox))
            sp <- TMFG(rdf, normal= FALSE)
    """
        # sp <- TMFG(rdf,na.data = "pairwise" ,normal = FALSE, depend=TRUE)
        print("got compute adj")
        result = conn.eval(test_r_script)
        tabmat = np.array(result["A"])
        tabmat = tabmat.squeeze()

        if ~conn.isClosed:
            conn.close()
    except Exception as e:
        conn.close()
        print(e.__doc__)
        print(str(e))
        return None

    return np.matrix(tabmat)


def compute_tfmg(file_path, r_port):
    conn = pyRserve.connect(port=r_port)
    print(conn)
    conn.r.file_path = file_path

    print("file path", file_path)

    test_r_script = r"""
            options(warn = -1)
            suppressPackageStartupMessages(library(NetworkToolbox))
            library(reticulate)
            np <- import("numpy")
            npz1 <- np$load(file_path)
            rdf <- npz1$f[["cor_mat"]]
            sp <- TMFG(rdf, normal= FALSE)
    """
    # sp <- TMFG(rdf,na.data = "pairwise" ,normal = FALSE, depend=TRUE)
    print("got compute adj")
    result = conn.eval(test_r_script)
    tabmat = np.array(result["A"])
    tabmat = tabmat.squeeze()

    if ~conn.isClosed:
        conn.close()

    return np.matrix(tabmat)


def distance_corr(df, filename=None):

    if isinstance(df, pd.DataFrame):
        rdf = df.values
    else:
        rdf = df

    col_size = rdf.shape[1]
    up_mat = np.zeros(shape=(col_size, col_size))
    for i in progressbar(range(1, col_size)):
        di = diag_off_indices(col_size, i)
        up_mat[di] = dcor.rowwise(
            dcor.distance_correlation,
            rdf[:, :-i].T,
            rdf[:, i:].T,
            compile_mode=dcor.CompileMode.COMPILE_PARALLEL,
        )

    cor_mat = complete_matrix(up_mat)
    if filename:
        np.savez(filename, cor_mat=cor_mat)

    return cor_mat


def diag_off_indices(col_size, offset):
    idx0 = np.arange(0, col_size - offset)
    idx = np.arange(offset, col_size)
    id_arr = (idx0, idx)

    return id_arr


def complete_matrix(X):
    return X + X.T - np.diag(np.diag(X))


def sharpe_ratio_loss(
    y_true,
    y_pred,
    annualize=True,
    trading_year_days=tf.constant(252.0, dtype=tf.float64),
):
    y_pred = tf.expand_dims(y_pred, axis=-1)
    # print(y_true.shape,y_pred.shape)

    # y_true = tf.cast(y_true,dtype=tf.float64)
    r = tf.matmul(y_true, y_pred)
    r = tf.reduce_sum(r, axis=-1)
    # sh_ratio = tf.reduce_mean(r) / tf.math.maximum(tf.math.reduce_std(r), 1e-3)
    sh_ratio = tf.reduce_mean(r) / tf.math.maximum(tf.math.reduce_std(r), 1e-3)
    if annualize:
        sh_ratio = sh_ratio * tf.math.sqrt(
            1.0 if trading_year_days is None else trading_year_days
        )
    return tf.reduce_mean(-sh_ratio)


class CustomSharpe(tf.keras.losses.Loss):
    def __init__(
        self,
        expand_dims=False,
        test=False,
        annualize=True,
        trading_year_days=tf.constant(252.0, dtype=tf.float32),
        name="custom_sharpe",
    ):
        super().__init__(name=name)
        self.expand_dims = expand_dims
        self.test = test
        self.annualize = annualize
        self.trading_year_days = trading_year_days

    def call(self, y_true, y_pred):
        if self.expand_dims:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        # y_true = tf.cast(y_true, dtype=tf.float32)
        # y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        r = tf.matmul(y_true, y_pred)

        r = tf.reduce_sum(r, axis=-1)
        r_add = tf.math.log(tf.add(r, 1))
        port_ret = tf.math.exp(tf.reduce_mean(r_add)) - 1
        # sh_ratio = tf.reduce_mean(r) / tf.math.maximum(tf.math.reduce_std(r), 1e-6)
        sh_ratio = port_ret / tf.math.maximum(tf.math.reduce_std(r), 1e-3)
        if self.annualize:
            sh_ratio = sh_ratio * tf.cast(
                tf.math.sqrt(
                    1.0 if self.trading_year_days is None else self.trading_year_days
                ),
                dtype=sh_ratio.dtype,
            )

        if self.test:
            # print("y_true:", y_true.shape, "y_pred: ", y_pred.shape)
            print("y_pred weights check: ", tf.reduce_sum(y_pred, axis=-1))
            # print(sh_ratio)
        return tf.reduce_mean(-sh_ratio)


class SharpeLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        expand_dims=False,
        test=False,
        annualize=False,
        trading_year_days=tf.constant(252.0, dtype=tf.float32),
        name="custom_sharpe",
    ):
        super().__init__(name=name)
        self.expand_dims = expand_dims
        self.test = test
        self.annualize = annualize
        self.trading_year_days = trading_year_days

    def call(self, y_true, y_pred):
        if self.expand_dims:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        # y_true = tf.cast(y_true, dtype=tf.float32)
        # y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        r = tf.matmul(y_true, y_pred)

        r = tf.reduce_sum(r, axis=-1)
        r_add = tf.math.log(tf.add(r, 1))
        port_ret = tf.math.exp(tf.reduce_mean(r_add)) - 1
        # sh_ratio = tf.reduce_mean(r) / tf.math.maximum(tf.math.reduce_std(r), 1e-6)
        sh_ratio = tf.math.log(port_ret) - tf.math.log(tf.math.reduce_std(r))
        if self.annualize:
            sh_ratio = sh_ratio * tf.cast(
                tf.math.sqrt(
                    1.0 if self.trading_year_days is None else self.trading_year_days
                ),
                dtype=sh_ratio.dtype,
            )

        if self.test:
            # print("y_true:", y_true.shape, "y_pred: ", y_pred.shape)
            print("y_pred weights check: ", tf.reduce_sum(y_pred, axis=-1))
            # print(sh_ratio)
        return tf.reduce_mean(-sh_ratio)


def plt_corr_nx(H, title, filename=None, graph_type="CONNECTED", pos=None):

    # creates a set of tuples: the edges of G and their corresponding weights
    edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())

    if pos is None:
        pos = nx.kamada_kawai_layout(H)

    # pos = nx.fruchterman_reingold_layout(H)
    scaling_factor = 2000

    with sns.axes_style("whitegrid"):
        # figure size and style
        plt.figure(figsize=(100, 45))
        plt.title(title, size=16)

        # computes the degree (number of connections) of each node
        deg = H.degree

        # list of node names
        if graph_type == "CONNECTED":
            nodelist, node_sizes = zip(*nx.get_node_attributes(H, "To").items())
        else:
            nodelist, node_sizes = zip(H.nodes, [0.002] * len(H.nodes))
        pos = nx.spring_layout(H, fixed=nodelist, pos=pos)

        # iterates over deg and appends the node names and degrees
        # for n, d in deg:
        #    nodelist.append(n)
        # node_sizes.append(nodedf.loc[n]['Weight']*scaling_factor)

        # draw nodes
        nx.draw_networkx_nodes(
            H,
            pos,
            # node_color="#DA70D6",
            node_color=node_sizes,
            cmap=plt.cm.Blues,
            nodelist=nodelist,
            node_size=np.array(node_sizes) * scaling_factor,
            alpha=0.7,
            # font_weight="bold",
        )

        # node label styles
        nx.draw_networkx_labels(
            H, pos, font_size=13, font_family="sans-serif", font_weight="bold"
        )

        # color map
        cmap = sns.cubehelix_palette(3, as_cmap=True, reverse=True)

        # draw edges
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=edges,
            style="solid",
            edge_color=weights,
            edge_cmap=cmap,
            edge_vmin=min(weights),
            edge_vmax=max(weights),
            arrowsize=4,
        )

        # builds a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Blues,
            norm=plt.Normalize(
                vmin=min(node_sizes), vmax=max(node_sizes) / scaling_factor
            ),
        )
        sm._A = []
        plt.colorbar(sm)

        # displays network without axes
        plt.axis("off")
        if filename:
            plt.savefig(filename)
            plt.close()

        return pos


def to_max_batch(n_max=None, x_list=None, a_list=None, e_list=None, mask=False):
    """
    Converts lists of node features, adjacency matrices and edge features to
    [batch mode](https://graphneural.network/data-modes/#batch-mode),
    by zero-padding all tensors to have the same node dimension `n_max`.
    Either the node features or the adjacency matrices must be provided as input.
    The i-th element of each list must be associated with the i-th graph.
    If `a_list` contains sparse matrices, they will be converted to dense
    np.arrays.
    The edge attributes of a graph can be represented as
    - a dense array of shape `(n_nodes, n_nodes, n_edge_features)`;
    - a sparse edge list of shape `(n_edges, n_edge_features)`;
    and they will always be returned as dense arrays.
    :param x_list: a list of np.arrays of shape `(n_nodes, n_node_features)`
    -- note that `n_nodes` can change between graphs;
    :param a_list: a list of np.arrays or scipy.sparse matrices of shape
    `(n_nodes, n_nodes)`;
    :param e_list: a list of np.arrays of shape
    `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`;
    :param mask: bool, if True, node attributes will be extended with a binary mask that
    indicates valid nodes (the last feature of each node will be 1 if the node is valid
    and 0 otherwise). Use this flag in conjunction with layers.base.GraphMasking to
    start the propagation of masks in a model.
    :return: only if the corresponding list is given as input:
        -  `x`: np.array of shape `(batch, n_max, n_node_features)`;
        -  `a`: np.array of shape `(batch, n_max, n_max)`;
        -  `e`: np.array of shape `(batch, n_max, n_max, n_edge_features)`;
    """
    if a_list is None and x_list is None:
        raise ValueError("Need at least x_list or a_list")

    if n_max is None:
        n_max = max([x.shape[0] for x in (x_list if x_list is not None else a_list)])

    # Node features
    x_out = None
    if x_list is not None:
        if mask:
            x_list = [np.concatenate((x, np.ones((x.shape[0], 1))), -1) for x in x_list]
        x_out = pad_jagged_array(x_list, (n_max, -1))

    # Adjacency matrix
    a_out = None
    if a_list is not None:
        if hasattr(a_list[0], "toarray"):  # Convert sparse to dense
            a_list = [a.toarray() for a in a_list]
        a_out = pad_jagged_array(a_list, (n_max, n_max))

    # Edge attributes
    e_out = None
    if e_list is not None:
        if e_list[0].ndim == 2:  # Sparse to dense
            for i in range(len(a_list)):
                a, e = a_list[i], e_list[i]
                e_new = np.zeros(a.shape + e.shape[-1:])
                e_new[np.nonzero(a)] = e
                e_list[i] = e_new
        e_out = pad_jagged_array(e_list, (n_max, n_max, -1))

    return tuple(out for out in [x_out, a_out, e_out] if out is not None)
