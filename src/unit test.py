import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = qs.utils.download_returns("GOOG")
print(stock)
stock.to_csv("goog.csv")
# show sharpe ratio
qs.stats.sharpe(stock)

# or using extend_pandas() :)
print(stock.sharpe())
qs.plots.snapshot(stock, title="Performance")
