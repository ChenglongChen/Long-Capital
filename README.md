# Long-Capital: Quant Trading with Microsoft [Qlib](https://github.com/microsoft/qlib)

## Performance
|Method| Signal Model | Trading Strategy | IR |
| :--- | :------- |:------- | :-----------: |
| [Supervised Learning](examples/sl.ipynb) | LGBM | TopkDropoutStrategy | 1.602594 |
| [Reinforcement Learning](examples/rl.ipynb)| LGBM | TopkDropoutParamStrategy | **2.14316**|

## Learning Curve
![metappo](fig/TopkDropoutParamStrategy.png)

## Dependency
- My fork of [Qlib](https://github.com/microsoft/qlib): https://github.com/ChenglongChen/qlib
- You can get crowd source data from [investment_data](https://github.com/chenditc/investment_data)
