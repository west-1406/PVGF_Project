import argparse

from utils import GetDataset,PushPredictData


# 提供预测接口
def PVGF(stationId,startTime,step):
    '''
    输入期望的基站ID,起始时间戳,以及期望的预测步长,将预测值post发出,并返回状态码
        输入参数：
            stationId : 基站id
            startTime : 起始时间戳,毫秒
            step      : 预测的步长
            day       : 使用数据的长度
        输出参数:
            status    : 响应代码
    '''
    # 计算需要的时间段
    dataset = GetDataset(stationId,startTime,day=5)
    status = PushPredictData(stationId,startTime,step,dataset)
    print(status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stationId',type=int,default=98,help='基站ID')
    parser.add_argument('--startTime',type=int,default=1672491600000,help='预测起始时间')
    parser.add_argument('--step',type=int,default=4,help='步长')
    opt = parser.parse_args()

    stationId,startTime,step = opt.stationId,opt.startTime,opt.step
    PVGF(stationId,startTime,step)

