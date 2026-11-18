# graph_port

Code for the paper:

> Kamesh Korangi, Christophe Mues, Cristián Bravo (2024). Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks. *arXiv preprint arXiv:2407.15532*. <https://doi.org/10.48550/arXiv.2407.15532>

This repository contains the official implementation of the paper, which applies a **Graph Attention Network (GAT)** to portfolio optimisation in the U.S. mid-cap market. The GAT learns non-linear relationships between firms to construct investment portfolios. The method builds dynamic networks of firms from the **distance correlation** of their return volatilities, a measure that captures non-linear dependencies. It filters these dense networks with the **Triangulated Maximally Filtered Graph (TMFG)** algorithm, which keeps the most significant relationships and keeps the computation tractable. The GAT is trained end to end with a loss derived from the Sharpe ratio, so it optimises risk-adjusted returns directly. Tested on 30 years of data, the model consistently outperformed mean-variance, network-based and equal-weighted portfolio strategies. The code builds the graphs, trains the GAT and computes the benchmark portfolios.

## Requirements

Python 3.8 or later and the packages listed in `requirements.txt`. The versions used for the paper were not recorded.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Banking-Analytics-Lab/graph_port.git
    cd graph_port
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

The code also needs:

  * **R with a running Rserve server.** `src/utils.py` connects to R through `pyRserve`. The TMFG step uses the R packages `NetworkToolbox` and `reticulate`; the connectedness graph uses `frequencyConnectedness`, `zoo` and `BigVAR`.
  * **pandas below 2.0** for `src/single_trainer.py`, which calls `DataFrame.append`.

**Computational intensity.** The distance correlation and TMFG filtering steps are computationally expensive, especially for a large universe of firms. The original research used a High Performance Computing (HPC) cluster with GPUs, the IRIDIS facility at the University of Southampton. Rerunning the analysis on a standard machine may be very time-consuming. The time a full run took is not documented.

`sc_py` is the Slurm script used on that cluster. It loads CUDA 11.0, activates a conda environment named `tf_25cu`, and runs `src/main.py`. Before you use it elsewhere, set `CONDA_ENV_LIB` to the `lib` directory of the conda environment whose libraries the script adds to `LD_LIBRARY_PATH`, and adapt the partition, the module names and the `/local/software/...` paths to your cluster. The script also adds a local glibc build in `~/Downloads/glibc-2.30/build/math/` to `LD_LIBRARY_PATH`.

## Data

The analysis was performed on a comprehensive dataset of daily closing prices for all U.S. mid-cap companies listed between 1990 and 2021, covering a total of 16,793 firms over the period. The model uses a three-year rolling window, resulting in a dynamic universe of approximately 5,000 firms at any given time.

**Important Note**: Due to licensing and proprietary restrictions, the raw financial data used in this study is **not** included in this repository. Users must obtain their own historical price data to run the analysis. The code expects Excel files of daily closing prices, from which it computes daily returns and their rolling volatility. `data/README.md` gives the file formats and the configuration keys that point to each file.

**Data Formatting**: Ensure your custom data is formatted correctly before running the preprocessing scripts. The model expects time-series data of historical returns for each firm in the investment universe.

### Data Contact

For questions regarding the methodology and data structure, please contact the authors.

  * [**Cristián Bravo**](https://www.linkedin.com/in/cristianbravor/)
  * **Kamesh Korangi**

## Training

Every entry point is in `src/` and reads its settings with [Hydra](https://hydra.cc) from a `config/` directory at the root of the repository (`config_path="../config"`). **That directory is not in the repository.** The configuration files used for the paper are not documented, so you must write them first. The keys the code reads are named below and in `data/README.md`.

The Slurm script `sc_py` runs:

```bash
python src/main.py --config-name EP
```

`src/main.py` chooses what to run from `app_params.model`:

| `app_params.model` | What runs |
|---|---|
| `GRAPH_FROM_DIR` | Builds the graphs from the Excel files in `app_params.data_path`. With `data_params.build_corr: true` it writes the distance-correlation matrices to `app_params.corr_dir`. Otherwise it filters them with TMFG through Rserve (`data_params.graph_type: TMFG`, port `Rserve_port`), or builds a connectedness graph, and writes `graph_<i>.npz` to `app_params.graph_dir`. Each job processes the files with index `10*(job_id-1)` to `10*job_id-1`. |
| `GRAPH_FROM_FILE` | The same for one file (`build_graphdataset_fromfile` in `src/data_processing.py`). |
| `GNN` | The GAT model, `GNNModel`. |
| `NWX` | The network-based portfolio, `NetworkModel` in `src/models.py`. |
| `EFFICIENT_PORTFOLIO` | The mean-variance portfolio, `EffPorModel` in `src/models.py` (PyPortfolioOpt). |
| `EQUAL_PORTFOLIO` | The equal-weighted portfolio, `EquPorModel` in `src/models.py`. |
| `LSTM` | An LSTM model, `LSTMModel` in `src/models.py`. |

Two further entry points train the GAT on the graph files in `app_params.data_path`, with the Hydra configuration `trainer`:

```bash
python src/single_trainer.py   # one model per graph file
python src/trainer.py          # one model across batches of graph files
```

`src/test.py` is a stand-alone version of the batch trainer that reads the graphs from `GRAPH_DIR`.

### Environment variables

| Variable | Read by | Default | Use |
|---|---|---|---|
| `OUTPUT_DIR` | `src/single_trainer.py` | `.` | Where results and TensorBoard logs go (see Evaluation). Hydra may change the working directory to its run directory, so set an absolute path. |
| `GRAPH_DIR` | `src/test.py` | `data/graph_3y3m` | Directory of the `graph_<i>.npz` files. |
| `IMAGE_DIR` | `src/move_make.py` | `data/Graph4Data` | Directory of the `graphviz_<i>.png` frames that `src/move_make.py` joins into a video. |
| `CONDA_ENV_LIB` | `sc_py` | none | `lib` directory added to `LD_LIBRARY_PATH`. |

### Known issues in the released code

  * The modules import each other as the package `graph_port` (for example `import graph_port.utils as ut`), but they sit in `src/`. They import only if `src/` is importable under the name `graph_port`.
  * `src/main.py` imports `GNNModel` from `models`, but `GNNModel` is defined in `src/gnn_models.py` and `src/gnn_models_time.py`.
  * `build_graphdataset_fromfile` in `src/data_processing.py` (option `GRAPH_FROM_FILE`) uses the undefined names `source_file` and `filename`.

### Other files

`src/parallel_test.py` times the computation modes of `dcor`. `src/move_make.py` makes a video from graph images. `src/unit test.py` is a `quantstats` example that downloads returns for one ticker. `src/copil.py` is a text-processing exercise unrelated to the paper.

## Evaluation

`src/main.py` prints the Sharpe ratio on the training, validation and test sets for each model. `src/single_trainer.py` writes, under `OUTPUT_DIR`:

  * `results_pract/graph_results.csv`: the training, validation and test results for each graph;
  * `results_pract/wgts/wgts_<i>_.csv`: the portfolio weights per firm for each graph;
  * `logs/gradient_tape/<time>/`: TensorBoard logs, including the share of firms with zero weight and the maximum weight.

The directory `results_pract/wgts/` must exist before the run.

The commands that produce each table and figure of the paper are not documented.

## Pretrained models

Not released. The repository holds no trained weights, and the reason is not documented.

## Results

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Table 1 (portfolio performance) | Not documented | Not documented |
| Table 2 (turnover) | Not documented | Not documented |
| Figures 1 to 10 | Not documented | Not documented |

## Citation

This code is the official implementation for the following paper. Please cite our work if you use this code in your research.

Korangi, K., Mues, C. and Bravo, C. (2024). **Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks**. arXiv preprint arXiv:2407.15532. [https://arxiv.org/abs/2407.15532](https://arxiv.org/abs/2407.15532)

```bibtex
@article{korangi2024largescale,
  title   = {Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks},
  author  = {Korangi, Kamesh and Mues, Christophe and Bravo, Cristi{\'a}n},
  year    = {2024},
  journal = {arXiv preprint arXiv:2407.15532},
  doi     = {10.48550/arXiv.2407.15532},
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Authors

  * **Kamesh Korangi** - University of Southampton
  * **Christophe Mues** - University of Southampton
  * **Cristián Bravo** - The University of Western Ontario

You can find other work from the lab at [Banking-Analytics-Lab](https://github.com/Banking-Analytics-Lab).

## Licence

MIT. See [LICENSE](LICENSE).

## Acknowledgements

  * This work was supported by the Economic and Social Research Council [grant number ES/P000673/1].
  * The last author acknowledges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC) [Discovery Grant RGPIN-2020-07114]. This research was undertaken, in part, thanks to funding from the Canada Research Chairs program.
  * We acknowledge the use of the IRIDIS High Performance Computing Facility at the University of Southampton.
  * This project relies on several excellent open-source libraries, including [Spektral](https://github.com/danielegrattarola/spektral), [NetworkX](https://networkx.org/), and [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt).

## Contributing

We welcome contributions to this project. Please feel free to open an issue to report bugs or suggest improvements. If you would like to contribute code, please open a pull request.
