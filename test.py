import pandas as pd
from matplotlib import pyplot as plt
from PVGF_API import timeStamp

# 处理数据集（训练集）把时间转换为0-24
def train_time_change(file_path,save_path):
    datakey = ['date', 'latitude', 'longitude', 'humidity', 'vapor_pressure', 'temperature', 'precip', 'solar_radiation',
               'pressure', 'wind_speed', 'wind_direction_degree', 'feels_like', 'wind_scale', 'code', 'power']

    # file_path = r'D:\CentBrowser\Download\PVGF_Project-main (1)\PVGF_Project-main\dataset\last_data.csv'
    # save_path = r'D:\CentBrowser\Download\PVGF_Project-main (1)\PVGF_Project-main\dataset\last.csv'
    datas = pd.read_csv(file_path)
    print(datas)
    dataset = datas.to_dict(orient='records')
    print(dataset)
    output = []
    for data in dataset:
        buff = []
        for key in datakey:
            if key == 'date':
                buff.append(float(data[key].split()[1].split(':')[0]))
            else:
                # 仅测试使用，后续需要插值
                buff.append(float(data[key] if data[key] else 0))
        output.append(buff)

    output1 = pd.DataFrame(output)
    output1.to_csv(save_path,
                   header=['date', 'latitude', 'longitude', 'humidity', 'vapor_pressure', 'temperature', 'precip', 'solar_radiation',
               'pressure', 'wind_speed', 'wind_direction_degree', 'feels_like', 'wind_scale', 'code', 'power'])

# 时间间隔从5分钟转换为1小时
# dataset = pd.DataFrame(dataset)
# dataset.fillna(0, inplace=True)
# power = dataset.loc[:, 'power'].to_list()
# num = dataset.loc[0, 'power']
# for i in range(1, len(power)):
#     num += power[i]
#     if i % 12 != 0:
#         dataset.drop(index=i, axis=0, inplace=True)
#     if i % 12 == 0:
#         dataset.loc[i, 'power'] = num
#         num = 0
#
# dataset = dataset.to_dict(orient='records')

# split的切片运用 str.split()[1].split(':')[0]表示切片空格之后和：之前的内容  str.split(":")[1]表示切片第一个：和第二个之间的内容
# str='2023/1/20  1:35:00'
# print(float(int(str.split()[1].split(':')[0])+int(str.split(":")[1])/100))
# 输出为1.35

# 赵全营数据处理测试
def data_processing_zhaoquanying(filepath,savepath):
    # filepath = r'C:\Users\IU\Desktop\发电\赵全营数据\赵全营数据2.csv'
    # savepath = r'C:\Users\IU\Desktop\发电\赵全营数据\train_data2.csv'
    dataset = pd.read_csv(filepath, encoding="gbk")
    # 读取列名
    colum = list(dataset.columns.values)
    colum.remove('wea')
    colum.remove('wind')
    colum.remove('air_level')
    colum.remove('day_weather')
    colum.remove('night_weather')
    colum.remove('sunrise')
    colum.remove('sunset')
    datakey = colum
    dataset = dataset.to_dict(orient='records')
    output = []
    for data in dataset:
        buff = []
        for key in datakey:
            if key == 'data_date':
                buff.append(float(int(data[key].split()[1].split(':')[0]) + int(data[key].split(":")[1]) / 100))
            else:
                # 仅测试使用，后续需要插值
                buff.append(float(data[key] if data[key] else 0))
        output.append(buff)
    # 将power移到最后一列
    output1 = pd.DataFrame(output)
    # 保存为csv文件
    output1.to_csv(savepath, header=datakey)
