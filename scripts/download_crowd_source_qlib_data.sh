rm -rf qlib_bin.tar.gz
wget https://github.com/chenditc/investment_data/releases/download/2023-01-31/qlib_bin.tar.gz
rm -rf ~/.qlib/qlib_data/cn_data/*
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
