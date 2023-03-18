# Long-Capital
Quant Trading with Qlib

## Performance
|Method| Model | Trading Strategy | IR |
| :--- | :------- |:------- | :-----------: |
| [Supervised Learning](examples/sl.ipynb) | LGBM | TopkDropoutStrategy | 1.644155 |
| [Reinforcement Learning](examples/rl.ipynb)| EpisodeInformationRatioReward+MetaPPO | TopkDropoutDynamicStrategy | **2.143160**|

## Dependency
- My fork of [Qlib](https://github.com/microsoft/qlib): https://github.com/ChenglongChen/qlib
- You can get crowd source data from [investment_data](https://github.com/chenditc/investment_data)
