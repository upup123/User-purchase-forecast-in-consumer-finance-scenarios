# User-purchase-forecast-in-consumer-finance-scenarios
### 项目背景：比赛-2018消费金融场景下的用户购买预测
### 运行说明：
1. create_fea.py：  
输入：data下的原始数据；  
功能：进行特征工程；  
输出：2个dataframe:用于训练的train_fea_df和用于测试的test_fea_df.
2. model.py:  
输入是create_fea.py的2个dataframe;  
功能：利用2个模型：GradientBoostingClassifier和ExtraTreesClassifier进行blending;  
输出：result为对test_fea_df中的userid的购买预测概率.
注意：  
1.关于调参：blending中使用的模型没有经过调参，要想得到合适的结果可以调参，这部分大家可以自行尝试  
2.关于模型个数：model.py中使用的模型还可以选择其他分类模型。
