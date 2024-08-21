import math
import numpy as np
import pandas as pd
import os
from tqdm import tqdm   #process bar
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler



'''----------------------------参数设置-------------------------------------------'''
config = {
    #super parameters
    'seed' : 613334 ,
    'train_valid_ratio' : 0.90,
    'batch_size' : 256,
    'n_epochs' : 65,
    'learning_rate' : 0.0001,
    'model_path' : './model.ckpt',
    'seq_len' : 80,                     # 时间序列个数

    # 特征选择
    'select_all' :False,
    'x_features' : [1,5,6,10,11,12,13,14,15,16,17,18,19,20], #0是索引，1 是 timestamp
    #[1,5,6,10,11,12,13,14,15,16,17,18,19,20]表现良好
    'y_features' : [12],  #!!!这里的列索引是 selected 完后的索引（count from 0）

    # model parameters
    'rnn_layers': 1,
    'rnn_output_dim': 512,
    'dnn_layers' : 4,
    'dnn_hidden_dim': 512,

    'dropout_rate' : 0.35,
    'early_stop' : 10
}
'''------------------------------------------------------------------------------'''


'''----------------------------固定种子----------------------------------------------'''
def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
'''--------------------------------------------------------------------------------'''



'''----------------------------数据预处理---------------------------------------------'''
#split
def train_valid_split(data_set, valid_ratio):
    '''Split provided training data into training set and validation set based on time series order'''
    train_set_size = int(valid_ratio * len(data_set))
    valid_set_size = len(data_set) - train_set_size
    train_set = data_set[:train_set_size]
    valid_set = data_set[train_set_size:]
    return np.array(train_set), np.array(valid_set)

#选择特征
def select_feat(train_data, valid_data, test_data, select_all,features):
    '''Selects useful features to perform regression'''
    if select_all:
        feat_idx = list(range(1,train_data.shape[1]))
    else:
        feat_idx = features

    return train_data[:,feat_idx], valid_data[:,feat_idx], test_data[:,feat_idx]

def find_extreme_values(data):
    # 将数据转换为 float64 类型
    data = data.astype(np.float64)

    # 检查是否存在无穷大或极大值
    if np.isinf(data).any():
        print("Data contains infinity values at:")
        inf_indices = np.argwhere(np.isinf(data))
        for index in inf_indices:
            print(f"Index {tuple(index)}: {data[tuple(index)]}")

    # 检查是否存在超大值
    max_value = np.finfo(np.float64).max
    if (data > max_value).any():
        print(f"Data contains values larger than float64 max ({max_value}):")
        large_value_indices = np.argwhere(data > max_value)
        for index in large_value_indices:
            print(f"Index {tuple(index)}: {data[tuple(index)]}")

    print("Check complete.")

#把15min timestamp转化成每天的 15min 序列，比如00:00 -> 0,00:30 -> 2 ,尝试捕捉每日时间段的规律，但没鬼用
def convert_to_15min_label(timestamp):
    # 确保输入是 datetime 类型
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)

    # 计算一天中的秒数
    seconds_in_day = (timestamp.hour * 3600) + (timestamp.minute * 60) + timestamp.second
    # 计算对应的 15 分钟时间标签
    label = seconds_in_day // 900  # 900 秒 = 15 分钟
    return int(label)

# 1.转化成标准化数据 2.生成时间序列
def preprocess_data(data, seq_length,y_indices, is_test: bool):
    x = []
    y = []

    # 数据的第一列是时间列，转化成每一天的 15分钟标签
    time_column = pd.to_datetime(data[:, 0])  # 将时间列转换为 pandas datetime 类型
    time_series = pd.Series(time_column)  # 转换为 Series
    time_labels = time_series.apply(convert_to_15min_label).values.reshape(-1, 1)  # 计算 15 分钟标签
    features = data[:, 1:]  # 剩余的列为特征
    data_unscaled = np.hstack((time_labels, features))
    print(f'dataunscaled shape: {data_unscaled.shape}')

    # 对特征进行标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_unscaled)

    # 生成时间序列
    if not is_test:
        for i in range(len(data_scaled) - seq_length):
            # 构建时间序列的输入部分 x
            x.append(data_scaled[i:i + seq_length, :])
            # 构建时间序列的输出部分 y
            y.append(data_scaled[i + seq_length, y_indices])  # 选择 y 的特征列

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

    else:
        for i in range(len(data_scaled) - seq_length):
            # 构建时间序列的输入部分 x
            x.append(data_scaled[i:i + seq_length, :])

        x = np.array(x, dtype=np.float32)

        print(f'test_seq_shape: {x.shape}')

        return torch.tensor(x, dtype=torch.float32), scaler

class CyptoDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
'''--------------------------------------------------------------------------------'''




'''----------------------------神经网络----------------------------------------------'''
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.35):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, lstm_out):
        # 计算注意力权重
        attn_scores = self.attention_weights(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # 计算加权上下文向量
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector

class LSTM_Attention_DNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, dnn_layers, dnn_hidden_dim, dropout_rate, rnn_output_dim, rnn_layers):
        super(LSTM_Attention_DNN_Model, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_output_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Attention 层
        self.attention = Attention(rnn_output_dim)

        # FC层
        self.fc = nn.Sequential(
            BasicBlock(rnn_output_dim, dnn_hidden_dim, dropout_rate),
            *[BasicBlock(dnn_hidden_dim, dnn_hidden_dim, dropout_rate) for _ in range(dnn_layers)],
            nn.Linear(dnn_hidden_dim, output_dim)  # 输出层
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        out = self.fc(context_vector)

        return out
'''-----------------------------------------------------------------------------'''


'''----------------------------训练函数---------------------------------------------'''
def trainer(train_loader,valid_loader,model,config,device):
    criterion = nn.HuberLoss(delta=2.01)   # 数据被标准正态化后应该用 1.0，但是 2.0 效果更好

    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader),epochs=config['n_epochs'])

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs =  config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        # visualize training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.6f}, Valid loss: {mean_valid_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['model_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
'''--------------------------------------------------------------------------------'''


'''--------------------------------生成预测--------------------------------------'''
def predict_singlestep(test_loader, x_features,y_features,scaler_test, model, device,num_tests):
    model.eval()

    all_outputs = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            output = model(x)
            output = output.cpu().numpy()
            all_outputs.append(output)


    all_outputs = np.concatenate(all_outputs, axis=0)

    next_step = np.zeros((num_tests,len(x_features)),float)

    next_step[:, y_features] = all_outputs[:,:]
    next_step = scaler_test.inverse_transform(next_step)

    return next_step


# 计算预测回归和真实回归的准确率
def accuracy(real_returns, preds_returns):
    direction_correct = 0
    for i in range(len(real_returns)):
        if (preds_returns[i] >= 0 and real_returns[i] >= 0) or (preds_returns[i] < 0 and real_returns[i] < 0):
            direction_correct += 1

    print(f'predicted direction accuracy : {direction_correct * 100 / len(real_returns)} %')
    print(f'amount of preds: {len(real_returns)}')
'''--------------------------------------------------------------------------------'''


'''----------------------------准备工作----------------------------------------------'''
same_seed(config['seed'])

raw_data = pd.read_csv('/Users/mac/Desktop/QuantData/BTC-USDT/past301days_15m.csv').values
#raw_test_data = pd.read_csv('/Users/mac/Desktop/QuantData/BTC-USDT/past5days.csv').values

train_data, valid_data = train_valid_split(raw_data,config['train_valid_ratio'])
valid_data, raw_test_data = train_valid_split(valid_data,0.7)  # 直接用这个split函数划分测试集

train_data, valid_data, test_data = select_feat(train_data,valid_data,raw_test_data,select_all=config['select_all'],features=config['x_features'])

print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")

#find_extreme_values(train_data)

train_data_x, train_data_y, scaler_train = preprocess_data(train_data,config['seq_len'],y_indices=config['y_features'],is_test=False)
valid_data_x, valid_data_y, scaler_valid = preprocess_data(valid_data,config['seq_len'],y_indices=config['y_features'],is_test=False)
test_data_x, scaler_test= preprocess_data(test_data,config['seq_len'],y_indices=config['y_features'],is_test=True)

print(f'train_data(preprocessed) type: {type(train_data_x)}')

print(f"""train_data_x size: {train_data_x.shape}
train_data_y size: {train_data_y.shape}
valid_data_x size: {valid_data_x.shape}
valid_data_y size: {valid_data_y.shape}
test_data_x size:{test_data_x.shape}""")

train_dataset = CyptoDataset(train_data_x,train_data_y)
valid_dataset = CyptoDataset(valid_data_x,valid_data_y)
test_dataset = CyptoDataset(test_data_x,y=None)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,pin_memory=True)
'''-----------------------------------------------------------------------------'''


'''----------------------------开始训练----------------------------------------------'''
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE:{device}')

model = LSTM_Attention_DNN_Model(
    input_dim=train_data_x.shape[2],
    output_dim=train_data_y.shape[1],
    dnn_layers=config['dnn_layers'],
    dnn_hidden_dim=config['dnn_hidden_dim'],
    dropout_rate=config['dropout_rate'],
    rnn_output_dim=config['rnn_output_dim'],
    rnn_layers=config['rnn_layers']
).to(device)

trainer(train_loader,valid_loader,model,config,device)

model.load_state_dict(torch.load('./model.ckpt'))
model.to(device)

test_predictions = predict_singlestep(test_loader,config['x_features'],config['y_features'],scaler_test, model, device,test_data.shape[0]-config['seq_len'])

preds_returns = test_predictions[:,-1]

real_returns = test_data[:,-1]
real_returns = real_returns[config['seq_len']:]

accuracy(real_returns,preds_returns)

