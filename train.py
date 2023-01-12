import argparse

from keras import optimizers


from model.Model import LSTMnet
from utils import Processing_data

# LSTM的训练函数
def trainLSTM(opt):
    # 数据部分
    dataset_path = opt.DatasetPath
    input_seq_length = opt.InputSeqLength
    feature_size = opt.FeatureSize
    output_size = opt.OutputSize
    train_X, train_y, test_X, test_y, _, _ = Processing_data(dataset_path, input_seq_length, output_size)
    # 优化器
    adam = optimizers.Adam(lr=opt.lr)
    # 模型部分
    model = LSTMnet(input_seq_length, output_size*feature_size, feature_size)
    model.compile(loss='mse', optimizer=adam)
    # fit network
    history = model.fit(train_X, train_y, epochs=opt.epoch, batch_size=opt.BatchSize, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    loss = model.evaluate(test_X, test_y)
    print(f"训练集 loss:{history.history['loss'][-1]}\n测试集 loss:{loss}\n")
    model.save(f'{opt.SavePath}/{opt.epoch}.h5')

    
# Transform的训练函数
def trainTransform(opt):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SavePath', type=str, default='./checkpoints', help='训练权重默认保存路径')
    parser.add_argument('--save', type=bool, default=True, help='是否保存模型')
    parser.add_argument('--DatasetPath', type=str, default='./dataset/new_data5.csv', help='数据集路径')
    parser.add_argument('--epoch', type=int, default=200, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--BatchSize', type=int, default=512, help='批量大小')
    parser.add_argument('--InputSeqLength', type=int, default=120, help='输入序列的长度')
    parser.add_argument('--FeatureSize', type=int,default=13, help='特征尺寸')
    parser.add_argument('--OutputSize', type=int, default=1,help='输出尺寸')
    opt = parser.parse_args()
    trainLSTM(opt)
    # trainTransform(opt)