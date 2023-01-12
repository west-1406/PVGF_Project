import sys
import argparse
sys.path.append("./")

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

from utils import Processing_data

# LightGBM训练函数
def TrainLightGBM(dataset_path, input_seq_length, output_size):
    # 加载数据集
    train_X, train_y, test_X, test_y, _, _ = Processing_data(dataset_path, input_seq_length, output_size,model_type = 'LightGBM')

    # 模型训练
    gbm = MultiOutputRegressor(LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20))
    gbm.fit(train_X, train_y)

    # 模型存储
    joblib.dump(gbm, './checkpopint/LightGBM.pkl')
    # 模型加载
    gbm = joblib.load('./checkpopint/LightGBM.pkl')

    # 模型预测
    y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration_)
    print(y_pred)

    # # 网格搜索，参数优化
    # estimator = LGBMRegressor(num_leaves=31)
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20, 40]
    # }
    # gbm = GridSearchCV(estimator, param_grid)
    # gbm.fit(X_train, y_train)
    # print('Best parameters found by grid search are:', gbm.best_params_)

# LightGBM API接口
def LightGBMModel(input_data):
    model = joblib.load("./checkpoints/LightGBM.pkl")
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

    TrainLightGBM(dataset_path, input_seq_length, output_size)
