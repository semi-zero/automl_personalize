import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging
from collections import defaultdict
import pickle

#전처리 class
class Preprocessing:
    def __init__(self, log_name, data, item_id_var, user_id_var, event, user_var, item_var):
        self.df = data  
        self.item_id_var = item_id_var
        self.user_id_var = user_id_var
        self.event = event
        self.user_var = user_var
        self.item_var = item_var
        self.logger = logging.getLogger(log_name)
        
        #결측치 처리 먼저 진행
        self.df = self.na_preprocess(self.df)
        
        self.df, self.user_df, self.item_df, self.interaction_df = self.get_per_df()
        
        
        
        # 결측치 확인 및 처리
    def na_preprocess(self, df):
        
        self.logger.info('결측치 확인')        
        
        try:
            df.fillna('NaN', inplace=True)

        except:
            self.logger.exception('결측치 처리에 문제가 발생하였습니다')
            
        self.logger.info(f'결측치 처리 이후 데이터 구성: {df.shape[0]} 행, {df.shape[1]}열')                  
        
        return df
    
    
    def get_per_df(self):
        
        self.df[self.item_id_var] = self.df[self.item_id_var].astype(str)
        self.df[self.user_id_var] = self.df[self.user_id_var].astype(str)
        
        #user 데이터셋 생성
        user_df = self.df[[self.user_id_var]+self.user_var].drop_duplicates(self.user_id_var).reset_index(drop=True)

        #item 데이터셋 생성
        item_df = self.df[[self.item_id_var]+self.item_var].drop_duplicates(self.item_id_var).reset_index(drop=True)

        #interaction 데이터셋 생성
        interaction_df=self.df[[self.item_id_var, self.user_id_var, self.event]]#, timestamp]]
        if self.df[self.event].dtypes != float:
            interaction_df[self.event] = 5
        
        self.logger.info('전처리 완료')
        self.logger.info('\n')
        self.logger.info(self.df.head())
        
        
        
        return self.df, user_df, item_df, interaction_df
    
    