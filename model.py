# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import operator
def blending_train(X_dev,y_dev,X_test,test_userid,seed):
	np.random.seed(seed)
	print('准备模型')
	n_flods=5
	clfs = [
	GradientBoostingClassifier(n_estimators = 100),
	ExtraTreesClassifier(n_estimators = 200),
	]

	skf = KFold(n=X_dev.shape[0],n_folds=n_flods)
	blend_train = np.zeros((X_dev.shape[0],len(clfs)))
	blend_test = np.zeros((X_test.shape[0],len(clfs)))
	cv_res = np.zeros((len(clfs), len(skf)))

	'''1.stacking'''
	for j, clf in enumerate(clfs):
	    print('Training [{}]{}'.format(j,clf))
	    blend_test_j = np.zeros((X_test.shape[0], len(skf)))
	    xx_cv = []
	    xx_pre = []
	    xx_beat = {}
	    for i, (train_index, cv_index) in enumerate(skf):
	        X_train = X_dev[train_index]
	        y_train = y_dev[train_index]
	        X_cv = X_dev[cv_index]
	        y_cv = y_dev[cv_index]
	        
            model = clf.fit(X_train,y_train)
            one_res = model.predict_proba(X_cv)[:,1].reshape([-1,])

            score = roc_auc_score(y_cv,one_res)
            blend_train[cv_index, j] = one_res
            cv_res[j, i] = score
            print('cv AUC:{}'.format(score))
            test_pred = model.predict_proba(X_test)[:,1].reshape([-1,])
            blend_test_j[:,i] = test_pred

	    blend_test[:,j] = blend_test_j.mean(1)
    	print('clf_{},AUC mean={}(std:{})'.format(j,cv_res[j,].mean(), cv_res[j,].std()))
	'''2.Get result by using Ridge'''
	alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
	bclf = RidgeCV(alphas=alphas,normalize=True,cv=5)
	'''Note!'''
	bclf.fit(blend_train,y_dev)
	print('Ridge Best alpha=',bclf.alpha_)

	y_test_pred = bclf.predict(blend_test)
	print('y_test_pred:',y_test_pred)
	tmp_auc = {}
	for t in [x/50 for x in range(1,30)]:
	    temp = roc_auc_score(y_test,np.where(y_test_pred>t,1,0))
	    tmp_auc[t] = temp
	tmp_auc = sorted(tmp_auc.items(), key=operator.itemgetter(1),reverse=True)
	xx_beat = tmp_auc[0][0]
	score = roc_auc_score(y_test, np.where(y_test_pred>xx_beat,1,0))
	print('Ridge acc={}'.format(score))

	result = pd.DataFrame(test_userid,columns=['USRID'])
	result['RST'] = y_test_pred
	return result
if __name__=='__main__':
	train_fea_df = pd.read_csv('train_fea_df.csv')
	test_fea_df = pd.read_csv('test_fea_df.csv')
	
	col = [x for x in train_fea_df.columns if (x!='USRID') and (x!='FLAG')]
	seed = 42

	X_dev = train_fea_df[col].values
	y_dev = train_fea_df['FLAG']
	
    X_test = test_fea_df[col].values
    test_userid = test_fea_df.pop('USRID')
    result = blending_train(X_dev,y_dev,X_test,test_userid,seed)
