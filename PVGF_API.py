import argparse
import numpy as np
from utils import GetDataset,PushPredictData
import time 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 特征归一化
scaler = MinMaxScaler()
dataset = pd.read_csv('./dataset/dataset_hour.csv',usecols=range(1,18))
# 文本转换为数字(关键字缺失，目前留空)
# 测试专用
del dataset['wind_direction']
del dataset['text']
del dataset['latitude']
del dataset['longitude']
del dataset['power']
scaler.fit_transform(dataset)


length = 120
datakey = ['dataTime','humidity','vaporPressure','temperature','precip','solarRadiation',
              'pressure','windSpeed','windDirectionDegree','feelsLike','windScale','weatherCode','power']


def timeStamp(timeNum): 
    timeStamp = float(timeNum/1000) 
    timeArray = time.localtime(timeStamp) 
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray) 

    return float(otherStyleTime.split()[1].split(':')[0])


# 数据格式转换，数据接口对齐
def TransformData(dataset):
    '''数据接口对齐
       输入参数：
           dataset : 获取的数据源
       输出数据：
           output_data : 格式对齐的数据
    '''
    # 数据集插值对齐
    # delt_time = 3600000
    # for i in range(1,len(dataset)):
        # num = int(max((dataset[i]['dataTime']-dataset[i-1]['dataTime'])/delt_time,1) - 1)
        # for j in  range(num):
        #     data = {}
        #     for key in datakey:
        #         data[key] = float(dataset[i][key])+(j+1)/(num+1)*(float(dataset[i+1][key])-float(dataset[i][key]))
        #     dataset.insert(i+j+1,data)
    # 截取使用长度
    dataset = dataset[-length:]
    # 输出数据接口
    output = []
    for data in dataset:
        buff = []
        for key in datakey:
            if key == 'dataTime':
                buff.append(timeStamp(data[key]))
            else:
                # 仅测试使用，后续需要插值
                buff.append(float(data[key] if data[key] else 0))
        output.append(buff)
    buff1 = scaler.transform(np.array(output)[:,:-1])
    buff2 = (np.array(output)[:,-1]/400000).reshape(-1,1)
    output = np.concatenate((scaler.transform(np.array(output)[:,:-1]),(np.array(output)[:,-1]/400000).reshape(-1,1)),1)
    output = np.array([output])
    return {'LSTMnet':tf.convert_to_tensor(output)}


# 提供预测接口
def PVGF(stationId,startTime,step):
    '''
    输入期望的基站ID,起始时间戳,以及期望的预测步长,将预测值post发出,并返回状态码
        输入参数：
            stationId : 基站id
            startTime : 起始时间戳,毫秒
            step      : 预测的步长
        输出参数:
            status    : 响应代码
    '''
    # 计算需要的时间段
    dataset = GetDataset(stationId,startTime,day=8)
    output_data = TransformData(dataset)
    status = PushPredictData(stationId,startTime,step,output_data)
    print(status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationId',type=int,default=115,help='基站ID')
    parser.add_argument('--startTime',type=int,default=1673532584419,help='预测起始时间')
    parser.add_argument('--step',type=int,default=4,help='步长')
    opt = parser.parse_args()
    
    while(1):
        stationId,startTime,step = opt.stationId,opt.startTime,opt.step
        startTime = int(round(time.time() * 1000))
        PVGF(stationId,startTime,step)
        print(f'本次{startTime} 预测已推送,休眠60分钟!')
        time.sleep(3610)

