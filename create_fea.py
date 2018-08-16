# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import time

def static_func(arr):
    if len(arr)!=0:
        mean = arr.mean()
        std = arr.std()
        max = arr.max()
        min = arr.min()
        tail = arr[-1]
        n = len(arr)
        niu = 0.0
        niu2 = 0.0
        niu3 = 0.0
        niu4 = 0.0
        for a in arr:
            niu += a
            niu2 += a**2
            niu3 += a**3
        
        niu /= n
        niu2 /= n
        niu3 /= n
        for a in arr:
            a -= niu
            niu4 += a**4
        niu4/=n
        
        sigma = np.sqrt(niu2-niu*niu)
        #当方差为0时，分布式对称的，因此skew=0,kurt=0
        if sigma!=0:
            skew = (niu3 -3*niu*sigma**2-niu**3)/(sigma**3) # 偏度计算公式
            kurt=niu4/(sigma**4) # 峰度计算公式:下方为方差的平方即为标准差的四次方
        else:
            skew = 0
            kurt = 0   
    else:
        mean = 0
        std = 0
        max = 0
        min = 0
        tail = 0
        skew = 0
        kurt = 0
    return [mean, std, max, min, skew, kurt, tail]
'''
统计特征
'''
def get_day_fea(row):
    name = 'day'
    arr = row[name].values
    arr = label_begin_day-arr
    arr_1 = pd.Series(arr).diff(1)[1:].values
    fea1 = static_func(arr)
    fea2 = static_func(arr_1)
    
    arr = row[name].unique()
    arr = label_begin_day-arr
    arr_1 = pd.Series(arr).diff(1)[1:].values
    fea3 = static_func(arr)
    fea4 = static_func(arr_1)
    fea1.extend(fea2)
    fea1.extend(fea3)
    fea1.extend(fea4)
    
    return fea1
def get_hour_fea(row):
    name = 'hour'
    arr = row[name].values
    arr_1 = pd.Series(arr).diff(1)[1:].values
    fea1 = static_func(arr)
    fea2 = static_func(arr_1)
    
    arr = row[name].unique()
    arr_1 = pd.Series(arr).diff(1)[1:].values
    fea3 = static_func(arr)
    fea4 = static_func(arr_1)
    fea1.extend(fea2)
    fea1.extend(fea3)
    fea1.extend(fea4)
    return fea1
def get_min_fea(row):
    name = 'min'
    arr = row[name].values
    arr_1 = pd.Series(arr).diff(1)[1:].values
    fea1 = static_func(arr)
    fea2 = static_func(arr_1)
    
    arr = row[name].unique()
    arr_1 = pd.Series(arr).diff(1)[1:].values
    fea3 = static_func(arr)
    fea4 = static_func(arr_1)
    fea1.extend(fea2)
    fea1.extend(fea3)
    fea1.extend(fea4)
    return fea1
def get_day_ct(row):
    name = 'day'
    #统计连续几天点击总次数
    arr = row[name].values
    from collections import Counter
    arr = Counter(arr)
    keys = []
    for item in arr:
        keys.append(item)
    keys.sort()
    cn = 0
    fn = 0
    res = []
    count_arr = []
    for item in keys:
        count_arr.append(arr[item])
        if cn==0:
            start_day = item
            res_num = arr[item]
            cn = 1
        else:
            start_day+=1
            if item==start_day:
                res_num+=arr[item]
                fn = 1
            else:
                if fn==1:
                    res.append(res_num)
                fn=0
                start_day = item
                res_num=arr[item]
    if fn==1:
        res.append(res_num)
    if len(res)==0:
        fea = [0,0]+[0]*7
    else:
        fea = [len(res),max(res)]
        fea1 = static_func(np.array(res))
        fea.extend(fea1)
    fea2 = static_func(np.array(count_arr))
    fea.extend(fea2)
    return fea
def get_day_weight(row):
    name='day'
    arr = row[name].values
    from collections import Counter
    arr = Counter(arr)
    temp = []
    window = int(end_day-begin_day+1)
    for k in range(begin_day,end_day+1):
        temp.append(arr[k]*((k-begin_day+1)/window))
    fea = static_func(np.array(temp))
    return fea
def get_kind_fea(row):#test中不会有新的kind出现
    name='kind'
    arr = row[name].values
    from collections import Counter
    arr = Counter(arr)
    temp=[]
    for kind in kind_id:
        temp.append(arr[kind])
    fea1 = static_func(np.array(temp))
    #频次
    kind_sum = sum(temp)
    if kind_sum!=0:
        kind_div_sum = list(np.array(temp)/kind_sum)
    else:
        kind_div_sum = [0]*len(kind_id)
    fea2 = static_func(np.array(kind_div_sum))
    fea1.extend(temp)
    fea1.extend(fea2)
    fea1.extend(kind_div_sum)
    #加入kind的总个数
    fea1.extend([kind_sum])
    return fea1
def get_what_fea(row):
    name = 'what'
    arr = row[name].values
    from collections import Counter
    arr = Counter(arr)
    temp=[]
    for what in what_id:
        temp.append(arr[what])
    fea1 = static_func(np.array(temp))
    #频次
    what_sum = sum(temp)
    if what_sum!=0:
        what_div_sum = list(np.array(temp)/what_sum)
    else:
        what_div_sum = [0]*len(what_id)
    fea2 = static_func(np.array(what_div_sum))
    fea1.extend(temp)
    fea1.extend(fea2)
    fea1.extend(what_div_sum)
    #加入what的总个数
    fea1.extend([what_sum])
    return fea1
def get_how_fea(row):
    name='how'
    arr = row[name].values
    from collections import Counter
    arr = Counter(arr)
    temp=[]
    for how in how_id:
        temp.append(arr[how])
    fea1 = static_func(np.array(temp))
    how_sum = sum(temp)
    if how_sum!=0:
        how_div_sum = list(np.array(temp)/how_sum)
    else:
        how_div_sum = [0]*len(how_id)
    fea2 = static_func(np.array(how_div_sum))
    fea1.extend(temp)
    fea1.extend(fea2)
    fea1.extend(how_div_sum)
    #加入how的总个数
    fea1.extend([how_sum])
    return fea1
def get_type_fea(row):
    name = 'TCH_TYP'
    arr = row[name].values
    #统计总TCH_TYP 0 1 2的个数
    from collections import Counter
    arr = Counter(arr)
    temp = []
    for i in range(3):
        temp.append(arr[i])
    fea1 = static_func(np.array(temp))
    #统计总TCH_TYP 0 1 2占sum的频次
    type_sum = sum(temp)
    if type_sum!=0:
        type_div_sum = list(np.array(temp)/type_sum)
    else:
        type_div_sum = [0]*3
    fea2 = static_func(np.array(type_div_sum))
    fea1.extend(temp)
    fea1.extend(fea2)
    fea1.extend(type_div_sum)
    return fea1
'''
时差特征：用户与商品kind-what交互的时间距离
'''
def func1(row):
    arr = list(map(lambda x,y:int(''.join([str(x),str(y)])),row['hour'].values,row['min'].values))
    fea = max(arr)-min(arr)
    return fea
def get_kind_hour_fea(row):
    temp = row.groupby(['USRID','day','kind']).apply(func1)
    arr = list(map(lambda x: x, temp.values))
    diff_sum = sum(arr)
    fea = static_func(np.array(arr))
    fea.extend([diff_sum])
    return fea
def get_kind_what_hour_fea(row):
    temp = row.groupby(['USRID','day','kind','what']).apply(func1)
    arr = list(map(lambda x: x, temp.values))
    diff_sum = sum(arr)
    fea = static_func(np.array(arr))
    fea.extend([diff_sum])
    return fea
def get_kind_what_how_hour_fea(row):
    temp = row.groupby(['USRID','day','kind','what','how']).apply(func1)
    arr = list(map(lambda x: x, temp.values))
    diff_sum = sum(arr)
    fea = static_func(np.array(arr))
    fea.extend([diff_sum])
    return fea
def get_data_fea(df,log_df):
    
    '''
    统计特征
    '''
    cre_fea = log_df[['USRID','day']].groupby(['USRID']).apply(get_day_fea)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['day_mean','day_std','day_max','day_min','day_skew','day_kurt','day_tail',
                                           'day_diff_mean','day_diff_std','day_diff_max','day_diff_min','day_diff_skew','day_diff_kurt',
                                           'day_diff_tail','day_unique_mean','day_unique_std','day_unique_max','day_unique_min','day_unique_skew',
                                           'day_unique_kurt','day_unique_tail','day_unique_diff_mean','day_unique_diff_std',
                                            'day_unique_diff_max','day_unique_diff_min','day_unique_diff_skew','day_unique_diff_kurt',
                                            'day_unique_diff_tail'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','hour']].groupby(['USRID']).apply(get_hour_fea)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['hour_mean','hour_std','hour_max','hour_min','hour_skew','hour_kurt','hour_tail',
                                           'hour_diff_mean','hour_diff_std','hour_diff_max','hour_diff_min','hour_diff_skew','hour_diff_kurt',
                                           'hour_diff_tail','hour_unique_mean','hour_unique_std','hour_unique_max','hour_unique_min','hour_unique_skew',
                                           'hour_unique_kurt','hour_unique_tail','hour_unique_diff_mean','hour_unique_diff_std',
                                            'hour_unique_diff_max','hour_unique_diff_min','hour_unique_diff_skew','hour_unique_diff_kurt',
                                            'hour_unique_diff_tail'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','min']].groupby(['USRID']).apply(get_min_fea)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['min_mean','min_std','min_max','min_min','min_skew','min_kurt','min_tail',
                                           'min_diff_mean','min_diff_std','min_diff_max','min_diff_min','min_diff_skew','min_diff_kurt',
                                           'min_diff_tail','min_unique_mean','min_unique_std','min_unique_max','min_unique_min','min_unique_skew',
                                           'min_unique_kurt','min_unique_tail','min_unique_diff_mean','min_unique_diff_std',
                                            'min_unique_diff_max','min_unique_diff_min','min_unique_diff_skew','min_unique_diff_kurt',
                                            'min_unique_diff_tail'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','day']].groupby(['USRID']).apply(get_day_ct)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['len_ct','max_ct','ct_mean','ct_std','ct_max','ct_min','ct_skew','ct_kurt','ct_tail',
                                           'count_ct_mean','count_ct_std','count_ct_max','count_ct_min','count_ct_skew','count_ct_kurt',
                                            'count_ct_tail'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','day']].groupby(['USRID']).apply(get_day_weight)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['day_weight_mean','day_weight_std','day_weight_max','day_weight_min','day_weight_skew',
                                            'day_weight_kurt','day_weight_tail'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','kind']].groupby(['USRID']).apply(get_kind_fea)
    res = list(map(lambda x: x, cre_fea.values))
    kind_count_col = ['kind_{}_count'.format(i) for i in range(1,len(kind_id)+1)]
    kind_ratio_col = ['kind_{}_ratio'.format(i) for i in range(1,len(kind_id)+1)]
    cf= pd.DataFrame(np.array(res),columns=['kind_mean','kind_std','kind_max','kind_min','kind_skew','kind_kurt','kind_tail']+kind_count_col+
                                           ['kind_ratio_mean','kind_ratio_std','kind_ratio_max','kind_ratio_min','kind_ratio_skew',
                                            'kind_ratio_kurt','kind_ratio_tail']+kind_ratio_col+['kind_sum'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','what']].groupby(['USRID']).apply(get_what_fea)
    res = list(map(lambda x: x, cre_fea.values))
    what_count_col = ['what_{}_count'.format(i) for i in range(1,len(what_id)+1)]
    what_ratio_col = ['what_{}_ratio'.format(i) for i in range(1,len(what_id)+1)]
    cf= pd.DataFrame(np.array(res),columns=['what_mean','what_std','what_max','what_min','what_skew','what_kurt','what_tail']+what_count_col+
                                           ['what_ratio_mean','what_ratio_std','what_ratio_max','what_ratio_min','what_ratio_skew',
                                            'what_ratio_kurt','what_ratio_tail']+what_ratio_col+['what_sum'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','how']].groupby(['USRID']).apply(get_how_fea)
    res = list(map(lambda x: x, cre_fea.values))
    how_count_col = ['how_{}_count'.format(i) for i in range(1,len(how_id)+1)]
    how_ratio_col = ['how_{}_ratio'.format(i) for i in range(1,len(how_id)+1)]
    cf= pd.DataFrame(np.array(res),columns=['how_mean','how_std','how_max','how_min','how_skew','how_kurt','how_tail']+how_count_col+
                                           ['how_ratio_mean','how_ratio_std','how_ratio_max','how_ratio_min','how_ratio_skew',
                                            'how_ratio_kurt','how_ratio_tail']+how_ratio_col+['how_sum'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','TCH_TYP']].groupby(['USRID']).apply(get_type_fea)
    res = list(map(lambda x: x, cre_fea.values))
    type_count_col = ['type_{}_count'.format(i) for i in range(1,3+1)]
    type_ratio_col = ['type_{}_ratio'.format(i) for i in range(1,3+1)]
    cf= pd.DataFrame(np.array(res),columns=['type_mean','type_std','type_max','type_min','type_skew','type_kurt','type_tail']+type_count_col+
                                           ['type_ratio_mean','type_ratio_std','type_ratio_max','type_ratio_min','type_ratio_skew',
                                            'type_ratio_kurt','type_ratio_tail']+type_ratio_col)
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    print('1@data shape:',df.shape)
    '''
    kind,what,how会产生len(kind_id)+len(what_id)+len(how_id)个稀疏特征，
    这些稀疏特征用deep network先训练,生成几十个model，然后将这些model的输出结果concat作为deep learning feature
    '''
    '''
    时差特征:同1个item的停留时间
    '''
    cre_fea = log_df[['USRID','kind','day','hour','min']].groupby(['USRID']).apply(get_kind_hour_fea)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['kind_hour_mean','kind_hour_std','kind_hour_max','kind_hour_min','kind_hour_skew',
                                            'kind_hour_kurt','kind_hour_tail','kind_hour_diff_sum'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','kind','what','day','hour','min']].groupby(['USRID']).apply(get_kind_what_hour_fea)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['kind_what_hour_mean','kind_what_hour_std','kind_what_hour_max','kind_what_hour_min','kind_what_hour_skew',
                                            'kind_what_hour_kurt','kind_what_hour_tail','kind_what_hour_diff_sum'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    
    cre_fea = log_df[['USRID','kind','what','how','day','hour','min']].groupby(['USRID']).apply(get_kind_what_hour_fea)
    res = list(map(lambda x: x, cre_fea.values))
    cf= pd.DataFrame(np.array(res),columns=['kind_what_how_hour_mean','kind_what_how_hour_std','kind_what_how_hour_max','kind_what_how_hour_min','kind_what_how_hour_skew',
                                            'kind_what_how_hour_kurt','kind_what_how_hour_tail','kind_what_how_hour_diff_sum'])
    cf['USRID'] = cre_fea.index
    df = pd.merge(df, cf, on=['USRID'],how='left')
    print('2@data shape:',df.shape)
    df.replace(np.nan,0,inplace=True)
    return df
    
def get_dataset(train_agg,test_agg,train_log,test_log,train_flg):
	train = pd.merge(train_flg,train_agg,on='USRID',how='left')
    test = test_agg
    
    '''
    train_log中只有39028个用户的信息,而train_agg中有80000个用户的信息,
    因此train_log中用户信息不全
    '''
    log_df = list(map(lambda x : x.split('-'),train_log.EVT_LBL))
    log_df= pd.DataFrame(np.array(log_df),columns=['kind', 'what', 'how'])
    log_df['USRID'] = train_log['USRID']
    log_df['day'] =train_log.OCC_TIM.apply(lambda x: int(x[8:10]))
    log_df['hour'] = train_log.OCC_TIM.apply(lambda x: int(x[11:13]))
    log_df['min'] = train_log.OCC_TIM.apply(lambda x: int(x[14:16]))
    temp_df = train_log[['USRID','TCH_TYP']]
    log_df = pd.merge(log_df,temp_df, on='USRID',how='left')
    global kind_id,what_id,how_id,begin_day,end_day,label_begin_day,label_end_day
    kind_id = log_df.kind.unique()
    kind_id.sort()
    
    what_id = log_df.what.unique()
    what_id.sort()
    
    how_id = log_df.how.unique()
    how_id.sort()
    
    begin_day = 1
    end_day = 31
    label_begin_day = 32
    label_end_day = 38

    print('start create train data')
    now = datetime.datetime.now()
    res_train = get_data_fea(train,log_df)
    end = datetime.datetime.now()
    print('train data complete,@use_time:',(end-now).seconds,'@shape:',res_train.shape)
    ##############################################################################################
    log_df = list(map(lambda x : x.split('-'),test_log.EVT_LBL))
    log_df= pd.DataFrame(np.array(log_df),columns=['kind', 'what', 'how'])
    log_df['USRID'] = test_log['USRID']
    log_df['day'] =test_log.OCC_TIM.apply(lambda x: int(x[8:10]))
    log_df['hour'] = test_log.OCC_TIM.apply(lambda x: int(x[11:13]))
    log_df['min'] = test_log.OCC_TIM.apply(lambda x: int(x[14:16]))
    temp_df = test_log[['USRID','TCH_TYP']]
    log_df = pd.merge(log_df,temp_df, on='USRID',how='left')
    print('start create test data')
    now = datetime.datetime.now()
    res_test = get_data_fea(test,log_df)
    end = datetime.datetime.now()
    print('train data complete,@use_time:',(end-now).seconds,'@shape:',res_test.shape)
    return res_train, res_test

if __name__=='__main__':
    start_time =time.time()
    print('begin',time.asctime( time.localtime(start_time) ))
    '''1.load data'''
    # 读取个人信息
    train_agg = pd.read_csv('data/train_agg.csv',sep='\t')
    test_agg = pd.read_csv('data/test_agg.csv',sep='\t')
    # 日志信息
    train_log = pd.read_csv('data/train_log.csv',sep='\t')
    test_log = pd.read_csv('data/test_log.csv',sep='\t')
    # 用户唯一标识
    train_flg = pd.read_csv('data/train_flg.csv',sep='\t')
	'''2.create train and test datasets'''
	train_fea_df, test_fea_df = get_dataset(train_agg,test_agg,train_log,test_log,train_flg)
	train_fea_df.to_csv('train_fea_df.csv',index=False)
	test_fea_df.to_csv('test_fea_df.csv',index=False)
    