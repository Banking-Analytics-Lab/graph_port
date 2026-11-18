# Data

## Source and access

The paper uses the daily closing prices of all U.S. mid-cap companies listed between 1990 and 2021:
16,793 firms over the period. A three-year rolling window gives a universe of about 5,000 firms at
any time.

Due to licensing and proprietary restrictions, the raw financial data is **not** included in this
repository, and no data goes into git. Neither the repository nor the paper names the data provider
or states the licence terms of the data. Users must obtain their own historical price data. For
questions on the methodology and the data structure, contact the authors (see the main README).

## Where the code expects the files

All locations come from the Hydra configuration (see "Training" in the main README), except where
an environment variable is named.

| Step | Configuration key or variable | Files |
|---|---|---|
| Input prices | `app_params.data_path` | Excel files `*.xlsx` whose names carry an integer index as `_<i>_.xlsx`, read from sheet `Sheet1` (`src/data_processing.py`, `get_prep_data`, `file_key`) |
| Processed list | `data_params.processed_file_path` | An Excel file whose values list the indices already processed; these files are skipped |
| Distance correlation | `app_params.corr_dir` | `corr_<i>.npz`, one array `cor_mat` (`src/utils.py`, `distance_corr`) |
| Graphs | `app_params.graph_dir` | `graph_<i>.npz` with arrays `x`, `a`, `e`, `y`, `x_ret`, `x_tic`, `x_ind` |
| Training input | `app_params.data_path` (Hydra config `trainer`), or `GRAPH_DIR` for `src/test.py` (default `data/graph_3y3m`) | The `graph_<i>.npz` files, sorted by `<i>` |
| Graph images | `IMAGE_DIR` for `src/move_make.py` (default `data/Graph4Data`) | `graphviz_<i>.png` |

## Input format

`get_prep_data` in `src/data_processing.py` reads each Excel file in one of two layouts, set by
`data_params.pivot_type`:

- `pivot_type: false`: a wide table with a `Date` column and one column of closing prices per firm.
- `pivot_type: true`: a long table with the columns `Date`, `Symbol` and `close_price`, which the
  code pivots to the wide layout. It drops a fixed list of index symbols (`.BVLG`, `.FTMIB`,
  `.GSPTSE`, `.OMXC20`, `.OMXHPI`, `.OMXSPI`, `.OSEAX`, `.SMSI`) if they are present.

From the prices the code computes daily returns and a 30-day rolling volatility, annualised with
252 trading days. The distance correlation is computed on the volatilities.
