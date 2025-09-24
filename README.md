# Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks

This repository contains the official implementation for the paper "Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks." The project introduces a novel deep learning approach for financial portfolio optimization, specifically targeting the challenging U.S. mid-cap market.

The core of this work is a **Graph Attention Network (GAT)** that learns complex, non-linear relationships between firms to construct high-performing investment portfolios. The methodology involves building dynamic networks of firms based on **distance correlation**—a measure that captures non-linear dependencies in their return volatilities. These dense networks are then filtered using the **Triangulated Maximally Filtered Graph (TMFG)** algorithm to retain the most significant relationships while ensuring computational tractability. The GAT model is trained end-to-end with a custom loss function derived from the Sharpe ratio, enabling it to directly optimize for risk-adjusted returns. Our model was tested on 30 years of data and consistently outperformed traditional mean-variance, network-based, and equal-weighted portfolio strategies.

## Data

The analysis was performed on a comprehensive dataset of daily closing prices for all U.S. mid-cap companies listed between 1990 and 2021, covering a total of 16,793 firms over the period. The model uses a three-year rolling window, resulting in a dynamic universe of approximately 5,000 firms at any given time.

**Important Note**: Due to licensing and proprietary restrictions, the raw financial data used in this study is **not** included in this repository. Users must obtain their own historical price data to run the analysis. The code is structured to process data in a standard format (e.g., a time-series of daily returns for each firm).

### Data Contact

For questions regarding the methodology and data structure, please contact the authors.

  * [**Cristián Bravo**](https://www.linkedin.com/in/cristianbravor/)
  * [**Kamesh Korangi**](https://www.linkedin.com/in/cristianbravor/)

## Getting Started

The primary script to run the entire experimental pipeline is `main.py`. This script handles data preprocessing, network construction, model training, and evaluation.

To replicate the analysis, first set up the environment and then run the main script. You will need to modify `config.py` to point to your dataset location and adjust any model hyperparameters.

```bash
# Run the full pipeline: data prep, training, and evaluation
python main.py --config config.py
```

### Requirements

To run this project, you will need Python 3.8+ and the packages listed in `requirements.txt`. We recommend setting up a virtual environment.

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

## Publications

This code is the official implementation for the following paper. Please cite our work if you use this code in your research.

Korangi, K., Mues, C. and Bravo, C. (2025). **Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks**. arXiv preprint arXiv:2407.15532. [https://arxiv.org/abs/2407.15532](https://arxiv.org/abs/2407.15532)

## Contributing

We welcome contributions to this project. Please feel free to open an issue to report bugs or suggest improvements. If you would like to contribute code, please open a pull request.

## Notes

  * **Computational Intensity**: The distance correlation and TMFG filtering steps are computationally expensive, especially for a large universe of firms. The original research utilized a High Performance Computing (HPC) cluster with GPUs. Rerunning the analysis on a standard machine may be very time-consuming.
  * **Data Formatting**: Ensure your custom data is formatted correctly before running the preprocessing scripts. The model expects time-series data of historical returns for each firm in the investment universe.

## Authors

  * **Kamesh Korangi** - University of Southampton
  * **Christophe Mues** - University of Southampton
  * **Cristián Bravo** - The University of Western Ontario

You can find other work from the lab at [Banking-Analytics-Lab](https://github.com/Banking-Analytics-Lab).

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE](https://opensource.org/license/mit) file for details.

## Acknowledgments

  * This work was supported by the Economic and Social Research Council [grant number ES/P000673/1].
  * The last author acknowledges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC) [Discovery Grant RGPIN-2020-07114]. This research was undertaken, in part, thanks to funding from the Canada Research Chairs program.
  * We acknowledge the use of the IRIDIS High Performance Computing Facility at the University of Southampton.
  * This project relies on several excellent open-source libraries, including [Spektral](https://github.com/danielegrattarola/spektral), [NetworkX](https://networkx.org/), and [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt).
