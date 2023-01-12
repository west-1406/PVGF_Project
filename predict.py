from model.Model import XGBoostModel,LSTMnet
from tensorflow.keras.models import load_model

LSTMnetFunc = load_model('./checkpoints/400.h5')
model_type = {'LSTMnet':LSTMnetFunc}

# 数据滑动更新
def dataUpdate(dataset,update):
    # 更新各个模型的数据集
    for key in model_type.keys():
        length = update[key]
        dataset[key] = dataset[key][length:]+update
    return dataset

def PredictPower(dataset,step):
    # 各个模型的电量预测结果
    power = {key:[] for key in model_type.keys()}
    # 模型预测
    for i in range(step):
        # 各个模型需要的更新的预测数据
        update = {}
        for key in model_type.keys():
            result = model_type[key](dataset[key])
            update[key] = result
            power[key].append(result[-1])
        # 滑动更新数据
        dataset = dataUpdate(dataset,update)
    # 算法测试
    length = len(model_type)
    result = [0 for i in range(length)]
    count = 0
    for key,value in power.items():
        result[count] += 1/length * value
        count += 1
    return result

