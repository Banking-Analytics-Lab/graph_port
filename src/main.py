import sys, os
import warnings
import numpy as np
import logging
import hydra
import tensorflow as tf

from omegaconf import DictConfig

import data_processing as dp
from models import GNNModel, LSTMModel, EffPorModel, EquPorModel, NetworkModel

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(
    suppress=True, threshold=sys.maxsize
)  # You can also use np.inf, but only np.inf can be used in tensorflow


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_floatx("float32")


log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="defaults")
def main(cfg: DictConfig) -> None:

    print("Config for ", cfg.name, " loaded")

    nm = None
    if cfg.app_params.model == "GNN":
        nm = GNNModel(cfg)
    elif cfg.app_params.model == "LSTM":
        nm = LSTMModel(cfg)
    elif cfg.app_params.model == "EFFICIENT_PORTFOLIO":
        nm = EffPorModel(cfg)
    elif cfg.app_params.model == "EQUAL_PORTFOLIO":
        nm = EquPorModel(cfg)
    elif cfg.app_params.model == "NWX":
        nm = NetworkModel(cfg)
    elif cfg.app_params.model == "GRAPH_FROM_FILE":
        # dp.build_graphdataset(cfg.app_params.data_path,cfg.app_params.graph_dir,cfg.data_params)
        dp.build_graphdataset_fromfile(cfg)
    elif cfg.app_params.model == "GRAPH_FROM_DIR":
        dp.build_graphdataset_fromdir(cfg)

    if nm is not None:
        nm.train_model(
            cfg.model_params.batch_size,
            cfg.model_params.epochs,
            cfg.model_params.patience,
        )


if __name__ == "__main__":
    main()
