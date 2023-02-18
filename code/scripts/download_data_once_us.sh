# 1. download csv
python3.8 ~/study/qlib/scripts/data_collector/yahoo/collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data --region US --interval 1d --max_workers 1 --delay 0 --max_collector_count 2

# 2. normalize csv
python3.8 ~/study/qlib/scripts/data_collector/yahoo/collector.py normalize_data --source_dir ~/.qlib/stock_data/source/us_data --normalize_dir ~/.qlib/stock_data/source/us_1d_nor --region US --interval 1d --max_workers 8

# 3. dump bin
python3.8 ~/study/qlib/scripts/dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/us_1d_nor --qlib_dir ~/.qlib/qlib_data/us_data --freq day --exclude_fields date,symbol --max_workers 8

# 4. prepare benchmark index
# parse instruments, using in qlib/instruments.
python3.8 ~/study/qlib/scripts/data_collector/us_index/collector.py --index_name SP400 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments
python3.8 ~/study/qlib/scripts/data_collector/us_index/collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments
python3.8 ~/study/qlib/scripts/data_collector/us_index/collector.py --index_name NASDAQ100 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments
python3.8 ~/study/qlib/scripts/data_collector/us_index/collector.py --index_name DJIA --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments
