import sys
import argparse
sys.path.append("./")

import joblib
from sklearn.multioutput import MultiOutputRegressor
import xgboost  as xgb
import pandas as pd

from utils import Processing_data

# XGBoost训练函数
def TrainXGBoost(dataset_path, input_seq_length, output_size):

    # 加载数据集
    train_X, train_y, test_X, test_y, data_x, dataset = Processing_data(dataset_path, input_seq_length, output_size,model_type = 'XGBoost')
    
    # 参数设定
    other_params = {
        'eta':0.3,
        'min_child_weight':1,
        'max_depth': 10, 
        'learning_rate': 0.1, 
        'gamma': 0, 
        'max_delta_step':0.2,
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'colsample_bylevel':1,
        'lambda':1,
        'reg_alpha': 0, 
        'reg_lambda': 1,
        'n_estimators': 800, 
        'seed': 0, 
    }

    # XGBoost训练
    model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',**other_params))

    model.fit(train_X, train_y)

    # 保存模型
    joblib.dump(model, "./checkpoints/XGBoost.pkl")

    # 训练集结果
    train_loss = train_y-model.predict(train_X)
    data = pd.DataFrame(train_loss)
    data.to_excel("train_loss.xlsx")

    # 测试集结果
    test_loss = test_y-model.predict(test_X)
    data.to_excel("test_loss.xlsx")
    print(sum(abs(train_loss[:,-1]))/len(train_loss[:,-1])*10000)
    print(sum(abs(test_loss[:,-1]))/len(test_loss[:,-1])*10000)

# 测试端口
def TestXGBoost(dataset_path, input_seq_length, output_size):
    model = joblib.load("./checkpoints/XGBoost.pkl")

    # 加载数据集
    train_X, train_y, test_X, test_y, data_x, dataset = Processing_data(dataset_path, input_seq_length, output_size,model_type = 'XGBoost')

    # 训练集结果
    train_loss = train_y-model.predict(train_X)
    print(sum(abs(train_loss[:,-1]))/len(train_loss[:,-1])*10000)
    # 测试集结果
    test_loss = test_y-model.predict(test_X)
    print(sum(abs(test_loss[:,-1]))/len(test_loss[:,-1])*10000)
    
     

    


if __name__ == '__main__':
    # 训练函数
    parser = argparse.ArgumentParser()
    parser.add_argument('--DatasetPath', type=str, default='./dataset/real_data.csv', help='数据集路径')
    parser.add_argument('--InputSeqLength', type=int, default=288, help='输入序列的长度')
    parser.add_argument('--OutputSize', type=int, default=1,help='输出尺寸')
    opt = parser.parse_args()

    dataset_path = opt.DatasetPath
    input_seq_length = opt.InputSeqLength
    output_size = opt.OutputSize
    
    # TestXGBoost(dataset_path, input_seq_length, output_size)
    TrainXGBoost(dataset_path, input_seq_length, output_size)
