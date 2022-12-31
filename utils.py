import requests
import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 模型相关
from predict import PredictPower

# 滑动窗口数据
def time_series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    :param data:作为列表或2D NumPy数组的观察序列。需要。
    :param n_in:作为输入的滞后观察数(X)。值可以在[1..len(数据)]之间可选。默认为1。
    :param n_out:作为输出的观测数量(y)。值可以在[0..len(数据)]之间。可选的。默认为1。
    :param dropnan:Boolean是否删除具有NaN值的行。可选的。默认为True。
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    origNames = df.columns
    cols, names = list(), list()
    cols.append(df.shift(0))
    names += [('%s' % origNames[j]) for j in range(n_vars)]
    n_in = max(0, n_in)
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        time = '(t-%d)' % i
        cols.append(df.shift(i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    n_out = max(n_out, 0)
    # forecast sequence (t, t+1, ... t+n)
    for i in range(1, n_out + 1):
        time = '(t+%d)' % i
        cols.append(df.shift(-i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 读取处理数据
def Processing_data(filepath, n_steps_in, n_steps_out, scale=1000,model_type = None):
    # 读取数据
    dataset = pd.read_csv(filepath,parse_dates=['time'], index_col=['time'],
                          usecols=range(1,36))
    # 数据插值，补充空白值
    data1 = dataset.interpolate()
    # 数据归一化(除power外)
    colums = ['ws_10', 'ws_30', 'ws_50', 'ws_70', 'ws_90', 'ws_120', 'ws_150', 'ws_200',
              'wd_10', 'wd_30', 'wd_50', 'wd_70', 'wd_90', 'wd_120', 'wd_150', 'wd_200', 'tem_10', 'tem_30',
              'tem_50', 'tem_70', 'tem_90', 'tem_120', 'tem_150', 'tem_200', 'rhu_10', 'rhu_30', 'rhu_50',
              'rhu_70', 'rhu_90', 'rhu_120', 'rhu_150', 'rhu_200', 'sr']
    # 最大最小值归一化
    scaler = MinMaxScaler()
    for col in colums:
        data1[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))
    data1['power']=data1['power']/scale
    # 划分数据集
    processedData1 = time_series_to_supervised(data1, n_steps_in, n_steps_out)
    data_x = processedData1.loc[:, f'ws_10(t-{n_steps_in})':'power(t-1)']  # 输入序列的长度和数据
    data_y = processedData1.loc[:, 'ws_10':'power']  # 输出序列的长度和数据
    train_X1, test_X1, train_y, test_y = train_test_split(data_x.values, data_y.values, test_size=0.3, shuffle=False)
    # 对训练集和测试集升维，满足LSTM的输入维度
    if model_type != 'XGBoost' and model_type != 'LightGBM':
        train_X = train_X1.reshape((train_X1.shape[0], n_steps_in, data1.shape[1]))
        test_X = test_X1.reshape((test_X1.shape[0], n_steps_in, data1.shape[1]))
    else:
        train_X,test_X = train_X1,test_X1
    return train_X, train_y, test_X, test_y, data_x, data1

# 创建数据集
def BuildDateset(dataset,dataset_path):
    with open(dataset_path,'w+') as file:
        for data in dataset:
            for val in data.values():
                file.write(f'{val} ')
            file.write('\n')

# 获取数据集
def GetDataset(stationId,endTime,day=5,save=False):
    '''
        输入参数：
            ststionId : 采集数据的基站ID
            endTime : 起始时间戳,毫秒
            day       : 选择时间戳,毫秒
            save      : 是否保存数据,默认False
        输出参数：
            dataset   : 样本数据,dict
    '''
    startTime = endTime-24*60*60*1000*day
    data_url = f'https://power.real-smart.tech/api/system/data/station/forecast/material/recent?stationId={stationId}&startTime={startTime}&endTime={endTime}'
    headers = {
        'AuthSysCode':'RSPV',
        'Authorization':'YWTE#fqrof1rv9iwym037znn6mjciji4g1281'
    }
    # 获取响应的数据并解析
    response = requests.get(data_url,headers=headers)
    if response.status_code == 200:
        response_data = response.json()['data']
        dataset = list(filter(lambda d:d['solarRadiation'],response_data))
        # 写入数据集
        if save:
            BuildDateset(dataset=dataset,dataset_path=f'./dataset/{stationId}_{startTime}_{endTime}.txt')
        return dataset
    else:
        return None       

# 推送测试数据
def PushPredictData(stationId,startTime,step,dataset,timeGap=3600000):
    push_url = 'https://power.real-smart.tech/api/system/data/station/forecast/result/recent/notify'
    data1={
        "stationId":stationId,
        "startTime": startTime,
        "timeGap" : timeGap,
        "step": step,
        "data": PredictPower(dataset,step)
    }
    headers = {
        'AuthSysCode' : 'RSPV',
        'Authorization' : 'YWTE#fqrof1rv9iwym037znn6mjciji4g1281',
        'Content-Type':'application/json'
    }
    response=requests.post(push_url,data=json.dumps(data1),headers=headers)
    return response.text

# 

if __name__ == '__main__':
    # 样例测试
    PushPredictData(98,1672070400000,4)