# Ubiquant Market Prediction
Ubiquant Market Prediction 解决方案

## 简介 brief introduction
九坤投资（Ubiquant）是一家位于中国的领先的量化对冲基金。依靠数学和计算机科学方面的国际人才
以及尖端技术来推动金融市场的量化投资。在这次比赛中，参赛者将建立一个预测投资回报率的模型。
在历史价格上训练和测试你的算法。优秀的参赛作品将尽可能准确地解决这个现实世界的数据科学问
题。
竞赛类型：本次竞赛属于数据分析挖掘/金融量化交易，这边推荐使用的模型：LightGBM/TabNet
赛题数据：数据集包含了从数以千计的投资的真实历史数据中得出的特征。参赛者的挑战是预测与
做出交易决策有关的模糊指标的价值。这是一场依靠时间序列API的代码竞赛，以确保模型不会利
用逆向时间来获得出要预测的价格。
评估标准：皮尔逊相关系数 [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
推荐阅读 Kaggle内一篇EDA来获取一些预备知识：[EDA- target analysis | Kaggle](https://www.kaggle.com/code/lucamassaron/eda-target-analysis)
本次比赛LGBM Baseline 推荐：[Ubiquant LGBM Baseline | Kaggle](Ubiquant LGBM Baseline | Kaggle)

Ubiquant is a leading quantitative hedge fund based in China. Rely on international talent in mathematics and computer science
And cutting-edge technology to drive quantitative investment in financial markets. In this competition, entrants will build a model that predicts ROI.
Train and test your algorithms on historical prices. Good entries will address this real-world data science problem as accurately as possible
question.

Competition type: This competition belongs to data analysis and mining/financial quantitative trading, the recommended model here: LightGBM/TabNet
Challenge Data: The dataset contains features derived from real historical data on thousands of investments. The contestant's challenge is to predict and
The value of fuzzy metrics in relation to making trading decisions. It is a code race to rely on the time series API to ensure that the model does not
Use inverse time to get the price you want to predict.



## 最终方案 The final proposal
本次竞赛方案策略如下：
数据处理上，首先训练数据较大有18GB，我们使用 reduce_mem_usage ，将占用内存压缩至了3GB，并且将训练csv文件保存成pickle，读取速度仅为之前的1/10。
本次量化竞赛，主办方直接提供了300个经过处理的特征，所以为了防止过拟合，我们没有做任何特征工程，只是用了300个官方特征。
本次竞赛只采用了LightGBM模型，首先在TimeSeriesSplit 5折 （训练和验证 4:1）下使用Optuna进行调参，然后固定好参数后，是用全量数据训练。
本次竞赛采用的评价标准是 皮尔逊相关系数 （Pearson correlation coefficient），所以我们自定义了LGBM中的损失函数 pearsonr。
最终我们采用的模型是全量数据训练的LGBM，使用5个seed，训练出5个模型进行融合，减少了预测结果的波动。在Inference阶段，方案依旧会从后台读取官方提供的supplemental_train.csv，对Lgbm模型进行finetune，保证了实时的预测。
finetune 阶段，迭代次数从1900次减少至450，最终完成预测。

The strategy for this competition is as follows:
In terms of data processing, firstly, the training data is 18GB in size. We use reduce_mem_usage to compress the occupied memory to 3GB, and save the training csv file as pickle, and the reading speed is only 1/10 of the previous one.
In this quantitative competition, the organizer directly provided 300 processed features, so in order to prevent overfitting, we did not do any feature engineering, but only used 300 official features.
This competition only uses the LightGBM model. First, Optuna is used to adjust the parameters under the TimeSeriesSplit 50% discount (training and verification 4:1), and then after the parameters are fixed, it is trained with the full amount of data.
The evaluation standard used in this competition is Pearson correlation coefficient (Pearson correlation coefficient), so we customized the loss function pearsonr in LGBM.
In the end, the model we adopted was LGBM trained with full amount of data. Using 5 seeds, we trained 5 models for fusion, reducing the fluctuation of prediction results. In the Inference stage, the solution will still read the official supplemental_train.csv from the background, and finetune the Lgbm model to ensure real-time prediction.
In the finetune stage, the number of iterations is reduced from 1900 to 450, and the prediction is finally completed.

## Models及对应的特征工程 Models and corresponding feature engineering
因为历史上的加密货币价格是不保密的，所以本次竞赛的Public Leaderboard没有意义，我们以自己的Cross Validation 为准。
1. 用kaggle上公开的LightGBM Baseline训练模型
2. 利用Optuna调参，CV：0.192
3. 全量数据训练，CV：0.231
4. 5个seed融合，CV：0.29

Because the price of cryptocurrency in history is not confidential, the Public Leaderboard of this competition is meaningless, we will take our own Cross Validation as the standard.
1. Use the LightGBM Baseline public on kaggle to train the model
2. Using Optuna to adjust parameters, CV: 0.192
3. Full data training, CV: 0.231
4. 5 seeds fusion, CV: 0.29

## 代码集 codeSet
1. Ubiquant_LGBM_Train_Tuning.ipynb # 调参
2. Ubiquant_LGBM_Train.ipynb # 全量数据训练
3. Ubiquant_LGBM_Inference.ipynb # 补充数据训练和推理

## 数据集 dataSet
官网数据：[Ubiquant Market Prediction | Kaggle](https://www.kaggle.com/competitions/ubiquant-market-prediction/data)

## 总结 summary
竞赛是由Ubiquant举办的，参赛者将建立一个预测投资回报率的模型。本次竞赛中我们选择量化金融竞赛中最常用的LightGBM模型，该模型相对于传统机器学习模型有更强的学习能力，而相对于DNN模型又有更好的鲁棒性。特征上我们直接使用了官方提供的300个脱敏特征，在TimeSeriesSplit5折下使用Optuna进行了调参。
最后我们将参数用于全量数据训练，并且使用了5个seed做模型融合，保证模型的稳定性。
在比赛评估期中，我们的方案还会自动获取官方提供的补充数据集，对模型进行finetune训练，保证了模型的实时预测能力，最终我们取得奖牌。

The competition is run by Ubiquant, and entrants will build a model that predicts ROI. In this competition, we choose the most commonly used LightGBM model in quantitative financial competitions. Compared with traditional machine learning models, this model has stronger learning ability and better robustness than DNN models. In terms of features, we directly used the 300 desensitization features provided by the government, and used Optuna to adjust the parameters under the TimeSeriesSplit5 discount.
Finally, we use the parameters for full data training, and use 5 seeds for model fusion to ensure the stability of the model.
During the evaluation period of the competition, our solution will also automatically obtain the supplementary data set provided by the official, and perform finetune training on the model to ensure the real-time prediction ability of the model, and finally we won the medal.

