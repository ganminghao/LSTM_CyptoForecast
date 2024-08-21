# Introduction

This is a naive attempt to predict Bitcoin returns (15-minute intervals) using an LSTM-based model. After losing half of my investment in the cryptocurrency market, I decided to try turning losses into profits by using a machine learning model. This is a completely new field for me (my experience so far has been limited to watching a few of Li Hongyi's YouTube videos). Thanks to the powerful Python libraries and ChatGPT, I managed to run the training code successfully. However, the model has proven ineffective in predicting returns.

# Get data 
By using API from OKX, I obtained basic BTC K-line features (Open, Close, High, Low) and trade volume in 15-minute intervals. I then generated SMA features, K-line-related features (Upper/Lower shadow, Range-to-change, Close-Open, High-Low, etc.), and 15-minute returns calculated recursively from closing prices. 

The data is written into a CSV file for further analysis.
（see get_past_data.py)

# Model design
LSTM layer + Attention layer + FC layer

# Training settings
Loss function: HuberLoss(Delta = 2.01)

Optimizer: AdamW

LearingRateScheduler: OneCycleLR(max_lr=0.01)

# Result
FAIL!

Best prediction accuracy: 52.4% (777 predictions) (Not within the 90% confidence interval)

# Reflection
Scenario 1: My model might be too simple and too easy to implement (otherwise, anyone who knows how to use Python packages could make a fortune). In the future, I may introduce more complex neural network structures or reconsider the feature selection based on financial theories.

Scenario 2: Past market data may not influence future trends: Fama (1970) established the Efficient Market Hypothesis (EMH), which assumes that market prices follow a random walk, meaning future price changes cannot be predicted using existing information. The EMH distinguishes three forms of market efficiency: weak-form, semi-strong form, and strong-form efficiency (Atsalakis and Valavanis, 2009a, Fama, 1970). (Reference: "Machine learning techniques and data for stock market forecasting: A literature review," Expert Systems with Applications, Volume 197, 1 July 2022, 116659). From this perspective, any crypto trader who claims to predict future trends solely from K-line patterns is either a fraud or simply riding a wave of luck—so far.



# 简介
这是一个基于 LSTM 模型的天真尝试，旨在预测比特币的回报（15 分钟间隔）。

在大一下半年， 我在 okx 痛失几百刀，于是这个暑假决定用机器学习模型来扭转亏损，妄想通过量化加密货币交易走上致富道路。感谢李宏毅教授的机器学习课程、感谢 ChatGPT、感谢强大的 python 库，最终写的代码能成功开始训练，但是训练出的模型在预测能力上不尽人意。

# 数据获取
通过 OKX API，获取基本的 BTC K 线特征（开盘价、收盘价、最高价、最低价）和 15 分钟间隔的交易量。然后生成了 SMA 特征、与 K 线相关的特征（上/下影线、Range-to-change、收盘-开盘价差、高低价差等），以及通过递归计算 15 分钟回报。写入 CSV 文件以供进一步分析。
（参见 get_past_data.py）

# 模型设计
LSTM layer + Attention layer + FC layer

# 训练设置
损失函数: Huber Loss（Delta = 2.01）

优化器: AdamW

学习率调度: OneCycleLR（max_lr=0.01）

# 结果
失败！

最佳预测准确率：52.4%（789 次预测）（不在 90% 置信区间内）

# 反思

情况 1： 我的模型可能太简单、门槛太低（不然随便一个会调用 python 库的人都能发财了）。以后可能会试试引入更复杂的神经网络结构，或者恶补金融学重新选择特征。

情况 2： 这个模型思路是完全不 work 的。市场的过去数据可能不会影响未来走势。从这个角度来看，任何声称仅通过 K 线趋势预测走势的币圈交易员要么是骗子，要么只是暂时跑赢了概率。“Fama（1970）提出了有效市场假说（EMH），该假说认为市场价格遵循随机游走，即未来的价格变化无法通过现有信息预测。EMH 区分了三种市场效率形式：弱式、半强式和强式效率（Atsalakis 和 Valavanis，2009a，Fama，1970）。（参考文献：《用于股票市场预测的机器学习技术和数据：文献综述》，Expert Systems with Applications, Volume 197, 2022 年 7 月 1 日, 116659）。”


# 欢迎大佬指点补充
# welcome to comment





