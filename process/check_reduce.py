import pandas as pd
import logging
import numpy as np

class Data_check_reduce:
    def __init__(self, log_name, data, item_id_var, user_id_var, event):
        self.logger = logging.getLogger(log_name)
        
        self.df = data
        self.item_id_var = item_id_var
        self.user_id_var = user_id_var
        self.event = event
        
        self.logger = logging.getLogger(log_name)
        
        #데이터 check가 통과되지 못하면 False로 변경
        self.check = True
        
        #데이터 check
        self.data_check()
       
        
    def data_check(self):
        
        self.logger.info('데이터 정합성 확인 절차 시작')        
        
        
        if self.df[self.item_id_var].isnull().sum() != 0:
            self.logger.info('상품 변수에 결측치가 포함되어 있습니다')
            self.check = False
        
        if self.df[self.user_id_var].isnull().sum() != 0:
            self.logger.info('사용자 변수에 결측치가 포함되어 있습니다')
            self.check = False
            
        if self.df[self.event].isnull().sum() != 0:
            self.logger.info('이벤트 변수에 결측치가 포함되어 있습니다')
            self.check = False
            
        
        self.logger.info('데이터 정합성 확인 절차 종료')
        self.logger.info(f'데이터 정합성 확인 절차 결과 : {self.check}')    
    
    
    