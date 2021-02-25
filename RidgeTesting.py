# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:33:57 2020

@author: Frankie
"""
# Machine Learning package for Ridge regression
from sklearn.linear_model import Ridge, RidgeCV

def cv_ridge(Xtr, Ytr, alphas, max_folds):
	if max_folds < 3:
		max_folds = 3
		
	for cvs in range(2, max_folds):
		best_ridge_regressor = RidgeCV(alphas=alphas, cv=cvs).fit(Xtr, Ytr)
		print("Best alpha for ", cvs, " folds: ", best_ridge_regressor.alpha_)
		
	print_ridge_info(best_ridge_regressor, Xtr, Ytr)
	return best_ridge_regressor

def linear_ridge(Xtr, Ytr, alpha):
	return Ridge(alpha=alpha).fit(Xtr, Ytr)

def linear_cv_ridge(Xtr, Ytr, alphas, folds=2):
	return RidgeCV(alphas=alphas, cv=folds).fit(Xtr, Ytr)

def print_ridge_info(regressor, Xtr, Ytr):
	print("Ridge regresson best params: ", regressor.alpha_)
	print("Chi-squared test against training data: ", regressor.score(Xtr, Ytr))
