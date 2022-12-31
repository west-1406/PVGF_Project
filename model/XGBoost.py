import sys
import argparse
sys.path.append("./")

import joblib
from sklearn.multioutput import MultiOutputRegressor
import xgboost  as xgb

from utils import Processing_data

# XGBoost训练函数
def TrainXGBoost(dataset_path, input_seq_length, output_size):
    # 加载数据集
    train_X, train_y, test_X, test_y, _, _ = Processing_data(dataset_path, input_seq_length, output_size,model_type = 'XGBoost')
    
    # 参数设定
    other_params = {
        'learning_rate': 0.1, 
        'n_estimators': 300, 
        'max_depth': 5, 
        'min_child_weight': 1,
        'seed': 0, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'gamma': 0, 
        'reg_alpha': 0, 
        'reg_lambda': 1
    }


    # XGBoost训练
    model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',**other_params))
    model.fit(train_X, train_y)

    # 保存模型
    joblib.dump(model, "./checkpoints/XGBoost.pkl")
    # 对测试集进行验证
    ans = model.predict(test_X)
    print(ans)

# XGBoost API接口
def XGBoostModel(input_data):
    model = joblib.load("./checkpoints/XGBoost.joblib.pkl")
    return model.predict(input_data)

if __name__ == '__main__':
    # 训练函数
    parser = argparse.ArgumentParser()
    parser.add_argument('--DatasetPath', type=str, default='./dataset/new_data5.csv', help='数据集路径')
    parser.add_argument('--InputSeqLength', type=int, default=288, help='输入序列的长度')
    parser.add_argument('--OutputSize', type=int, default=1,help='输出尺寸')
    opt = parser.parse_args()

    dataset_path = opt.DatasetPath
    input_seq_length = opt.InputSeqLength
    output_size = opt.OutputSize

    TrainXGBoost(dataset_path, input_seq_length, output_size)
