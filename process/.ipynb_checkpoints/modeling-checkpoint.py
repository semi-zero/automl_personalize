import pandas as pd
import numpy as np
import datetime

import tqdm
import math

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.spatial import distance
import matplotlib.pyplot as plt
import warnings

import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging

#surprise
from surprise import SVD, accuracy, SVDpp, KNNWithMeans, BaselineOnly
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

class Modeling:
    def __init__(self, log_name, df, user_df, item_df, interaction_df, item_id_var, user_id_var, event, num, model_type='auto'):
        self.df = df
        self.user_df = user_df
        self.item_df = item_df
        self.interaction_df = interaction_df
        self.item_id_var = item_id_var
        self.user_id_var = user_id_var
        self.event = event
        self.num = num
        self.model_type = model_type
        
        self.model = dict()
        self.score = dict()
        
        
        self.logger = logging.getLogger(log_name)
        
        self.start_time = datetime.datetime.now()
        
        # 결과값 딕셔너리
        self.score[f'Precision@{self.num}']     = dict()
        self.score[f'Recall@{self.num}']        = dict()
        self.score['RMSE']                      = dict()
        
        
        
        #모델링 딕셔너리
        model_type_dict = {'svd'  : self.svd_process(),
                           'pop'  : self.pop_process(),
                           'auto'  : self.auto_process()}
        
        #self.best_model_name, self.best_model, self.best_test = self.get_best_model()
       
        #학습 결과 화면을 위한 함수들
        #1. 분석 리포트
        #self.report = self.make_report(self.user_id_var, self.item_id_var, self.start_time)
        
        #2. 학습 결과 비교 화면
        #3. 변수 중요도
        #self.test_score, self.valid_score, self.fi = self.get_eval(self.best_model_name, self.best_model, self.best_test, self.ori_df, self.unique_id, self.target)
        
        
        #self.to_result_page()
        
        
        #################### SVD START####################
    
    def svd_process(self):
        if self.model_type == 'svd': 
            self.svd_fit_predict(self.interaction_df, self.item_df, self.item_id_var, self.user_id_var, self.event, self.num)
        
        
    #모델 fit
    def svd_fit_predict(self, interaction_df, item_df, item_id_var, user_id_var, event, num):
        
        #scoring 
        self.logger.info('svd 모델 fitting')
        
        def calculate_similarity(interaction_df, latent_features=50):
            pt = pd.pivot_table(index=user_id_var, columns=item_id_var, values=event, data=interaction_df, fill_value=0) #interaction데이터 pivot table 생성
            pt_m = pt.to_numpy()                                     # pivot table 생성
            user_ratings_mean = np.mean(pt_m, axis = 1)              # 사용자별 rating 평균 구하기
            df_demeaned = pt_m - user_ratings_mean.reshape(-1, 1)    # pivot table에서 rating 평균 빼기
            U, sigma, Vt = svds(df_demeaned, latent_features)        # SVD 분해  
            sigma = np.diag(sigma)                        
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            preds_df = pd.DataFrame(all_user_predicted_ratings, index=pt.index, columns = pt.columns)
            return preds_df
        
        preds_df = calculate_similarity(interaction_df)
        
        
        #아이템별 precision@k와 recall@k의 값을 알려줌
        def precision_recall(preds_df, item_df, interaction_df, num, user_id):

            def intersect(a, b):
                return list(set(a) & set(b))


            #유저 아이디 df
            user_id_df = interaction_df[interaction_df[user_id_var]==user_id]

            #유저 아이디가 본 아이텤 df
            user_item_df = item_df[item_df[item_id_var].isin(user_id_df[item_id_var].tolist())]
            user_item_list = user_item_df[item_id_var].tolist()

            #추천 아이템 추출
            sorted_user_predictions = pd.DataFrame(preds_df.loc[user_id, :].sort_values(ascending=False).reset_index())
            sorted_user_predictions.columns = [item_id_var, 'predictions']
            
            recommend_total_list_df = sorted_user_predictions.head(num)
            recommend_total_list = recommend_total_list_df[item_id_var].tolist()

            #순수 추천 아이템 리스트
            recommend_df_list = sorted_user_predictions[~sorted_user_predictions[item_id_var].isin(user_item_list)].head(num)[item_id_var].values.tolist()
            recommend_df = item_df[item_df[item_id_var].isin(recommend_df_list)]
            


            #지표 산정
            precision = len(intersect(user_item_list, recommend_total_list))/ len(recommend_total_list)
            recall = len(intersect(user_item_list, recommend_total_list)) / len(user_item_list)

            return precision, recall, user_item_df, recommend_df


        def perf_metric(preds_df, item_df, interaction_df, num):

            user_item_dfs = pd.DataFrame()
            recommend_dfs = pd.DataFrame()
            dict_list = []
            for USER_ID in tqdm.tqdm(interaction_df[user_id_var].unique()):
            #for USER_ID in tqdm.tqdm(['2']):
                precision, recall, user_item_df, recommend_df = precision_recall(preds_df, item_df, interaction_df, num, USER_ID)
                dict_ = {}
                dict_.update({"userID" : USER_ID})
                dict_.update({f"precision@{num}" : precision})
                dict_.update({f"recall@{num}" : recall})
                dict_list.append(dict_)

                #이미 user가 선정했던 item 확인
                user_item_df = user_item_df.copy()
                user_item_df.loc[:, user_id_var] = USER_ID
                user_item_dfs= pd.concat([user_item_dfs, user_item_df], axis=0)


                #상위 추천 item 중 이미 user가 선정했던 item 제외 후 추천 목록 추출
                recommend_df = recommend_df.copy()
                recommend_df.loc[:, user_id_var] = USER_ID
                recommend_dfs = pd.concat([recommend_dfs, recommend_df], axis=0)
            accuracy_df = pd.DataFrame(dict_list)
            return accuracy_df, user_item_dfs, recommend_dfs
        
        accuracy_df, user_item_dfs, recommend_dfs = perf_metric(preds_df, item_df, interaction_df, num)
        
        
        self.score[f'Precision@{self.num}']  = np.round(np.mean(accuracy_df[f'precision@{num}']),3)
        self.score[f'Recall@{self.num}']     = np.round(np.mean(accuracy_df[f'recall@{num}']),3)
        
        print(self.score)
        user_item_dfs.to_csv('storage/user_item_dfs.csv', index=False)
        recommend_dfs.to_csv('storage/recommend_dfs.csv', index=False)
        
        #################### SVD FINISH####################
        
        #################### POP START####################
        
    def pop_process(self):

        self.pop_fit_predict(self.interaction_df, self.item_df, self.item_id_var, self.user_id_var, self.event, self.num, self.df)


    #모델 fit
    def pop_fit_predict(self, interaction_df, item_df, item_id_var, user_id_var, event, num, df):

        def get_score(x):
            result = x['mean'] - (x['mean'] -0.5)*(math.pow(2, -np.log(x['count'])))
            return result

        if df[event].dtypes != float:
            pop_df = interaction_df.groupby(item_id_var)[event].agg(['count'])
            pop_df = pd.DataFrame(pop_df.reset_index())
            pop_list = pop_df.sample(n=num, weights='count')[item_id_var].tolist()

        else :
            pop_df = interaction_df.groupby(item_id_var)[event].agg(['mean', 'count'])
            pop_df = pd.DataFrame(pop_df.reset_index())
            pop_df['score'] = pop_df.apply(get_score, axis=1)
            pop_df = pop_df.sort_values('score', ascending=False)
            pop_df['weight'] = pop_df['score'].apply(math.exp)

            pop_list = pop_df.sample(n=num, weights='weight')[item_id_var].tolist()

        pop_recommend_df = item_df[item_df[item_id_var].isin(pop_list)]
        pop_recommend_df.to_csv('storage/pop_recommend_df.csv', index=False)
       
            
      #################### POP START####################
    
    
      #################### Auto START####################
        
    def auto_process(self):
        if self.model_type == 'auto': 
            self.auto_fit_predict(self.interaction_df, self.item_df, self.item_id_var, self.user_id_var, self.event, self.num)


    #모델 fit
    def auto_fit_predict(self, interaction_df, item_df, item_id_var, user_id_var, event, num):
        
        rating_min = 0.5
        rating_max = np.max(interaction_df[event].values)
        
        reader = Reader(rating_scale = (0, rating_max))
        df = Dataset.load_from_df(interaction_df[[user_id_var, item_id_var, event]], reader = reader)
        train, test = train_test_split(df, test_size = 0.2, random_state=42)

        algorithms = {'base': BaselineOnly()} 
                     #'knn' : KNNWithMeans(), 
                     #'svd' : SVD(), 
                     #'svdpp' :SVDpp()}
                    
        for algo_name, algo in zip(algorithms.keys(), algorithms.values()):
            algo.fit(train)
            predictions = algo.test(test)
            self.score['RMSE'][algo_name] = np.round(accuracy.rmse(predictions) , 3)
            self.model[algo_name] = algo
        
        best_model_name = min(self.score['RMSE'], key = self.score['RMSE'].get)
        best_model = self.model[best_model_name]
        
        def get_predictions(x):
            result = best_model.predict(x[user_id_var], x[item_id_var]).est
            return result

        def intersect(a, b):
            return list(set(a) & set(b))

        #아이템별 precision@k와 recall@k의 값을 알려줌
        def precision_recall(preds_df, item_df, interaction_df, num, user_id):

            #유저 아이디 df
            user_id_df = interaction_df[interaction_df[user_id_var]==user_id]

            #유저 아이디가 본 아이텤 df
            user_item_df = item_df[item_df[item_id_var].isin(user_id_df[item_id_var].tolist())]
            user_item_list = user_item_df[item_id_var].tolist()

            #총 추천 아이템 추출
            sorted_user_predictions = preds_df.sort_values('predictions',ascending=False)

            recommend_total_list_df = sorted_user_predictions.head(num)
            recommend_total_list = recommend_total_list_df[item_id_var].tolist()

            #순수 추천 아이템 리스트
            recommend_df_list = sorted_user_predictions[~sorted_user_predictions[item_id_var].isin(user_item_list)].head(num)[item_id_var].values.tolist()
            recommend_df = item_df[item_df[item_id_var].isin(recommend_df_list)]

            #지표 산정
            precision = len(intersect(user_item_list, recommend_total_list))/ len(recommend_total_list)
            recall = len(intersect(user_item_list, recommend_total_list)) / len(user_item_list)

            return precision, recall, user_item_df, recommend_df


        def perf_metric(user_df, item_df, interaction_df, num):

            user_item_dfs = pd.DataFrame()
            recommend_dfs = pd.DataFrame()
            dict_list = []
            for USER_ID in tqdm.tqdm(user_df[user_id_var].values):
            #for USER_ID in tqdm.tqdm(['1']): #,'2','3','4']):
                preds_df = item_df.copy()
                preds_df.loc[:, user_id_var] = USER_ID
                #preds_df 계산
                preds_df.loc[:, 'predictions'] = preds_df.apply(get_predictions, axis=1)
                precision, recall, user_item_df, recommend_df = precision_recall(preds_df, item_df, interaction_df, num, USER_ID)
                dict_ = {}
                dict_.update({"userID" : USER_ID})
                dict_.update({f"precision@{num}" : precision})
                dict_.update({f"recall@{num}" : recall})
                dict_list.append(dict_)

                #이미 user가 선정했던 item 확인
                user_item_df = user_item_df.copy()
                user_item_df.loc[:, user_id_var] = USER_ID
                user_item_dfs= pd.concat([user_item_dfs, user_item_df], axis=0)


                #상위 추천 item 중 이미 user가 선정했던 item 제외 후 추천 목록 추출
                recommend_df = recommend_df.copy()
                recommend_df.loc[:, user_id_var] = USER_ID
                recommend_dfs = pd.concat([recommend_dfs, recommend_df], axis=0)
            accuracy_df = pd.DataFrame(dict_list)
            return accuracy_df, user_item_dfs, recommend_dfs
        accuracy_df, user_item_df, recommendations_df = perf_metric(user_df, item_df, interaction_df, num = 10) 

        self.score[f'Precision@{self.num}']  = np.round(np.mean(accuracy_df[f'precision@{num}']),3)
        self.score[f'Recall@{self.num}']     = np.round(np.mean(accuracy_df[f'recall@{num}']),3)
        
        print(self.score)
        user_item_dfs.to_csv('storage/user_item_dfs.csv', index=False)
        recommend_dfs.to_csv('storage/recommend_dfs.csv', index=False)
    #################### Auto Finish####################    
        
    
    #def make_report(self, target, model_type, over_sampling, hpo, start_time):
    def make_report(self, user_id_var, item_id_var, start_time):
                    
        self.logger.info('학습 결과를 위한 결과물 생성')
        try:
            report = pd.DataFrame({'상태' : '완료됨',
                                  '모델 ID' : ['model_id'],
                                  '생성 시각': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
                                  '학습 시간' : [datetime.datetime.now()-start_time],
                                   '데이터셋 ID' : 'dataset_id',
                                   '사용자 변수' : user_id_var,
                                   '상품 변수' : item_id_var,
                                   '데이터 분할' : '80/20',
                                   '알고리즘' : model_type, 
                                   '목표' : '추천',
                                   '최적화 목표' : 'RMSE',
                                   '불균형 처리 여부' : over_sampling,
                                   'HPO 여부' : hpo})
                                    #hpo여부도 필요
            report = report.T
        
        except:
            self.logger.exception('학습 결과를 위한 결과물 생성 실패했습니다')
            
        return report