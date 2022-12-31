# PVGF_Project
短时光伏发电预测工程文件

## 更新时间 2022.12.30

1.初始化仓库，添加训练函数以及样例数据\
2.函数封装，接口封装\
3.新增数据接收函数

## 算法整体框架图
![farm](./material/frame.png)

## 特征异常值检测
绘制各个特征以及对应发电量随时间变化的曲线图，寻找异常点的数据特征，使用插值法或者其他方式去除异常值

## 特征相关性分析
计算各特征与发电量的皮尔森相关系数，在同一张图上绘制特征量、发电量随时间变化的曲线图，确定具有关键特征

## 高阶特征环境分析
选择相关度较高的特征，进行组合变换，生成高阶特征，再判断高阶特征与发电量的变化关系

## 多模型融合预测
经过高阶特征分析后，特征维度可以提升，选择多个不同的模型（目前可以考虑使用LSTM,TransForm,XGBoost,LightGBM），对应输入不同的特征，进行发电量预测，最终计算加权平均值作为最终输出值

### 滑动预测法
使用前一段时间的数据预测后一个时间点的所有特征，下次的预测值将使用本次预测值作为输入，但是**存在累计误差**，目前解决办法是**通过寻找前面一段时间内相似的数据，作为额外的输入特征，对输入维度进行concat，期望降低预测带来的累计误差**
