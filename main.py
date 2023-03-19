import argparse
import logging
import pandas as pd
import random
import os
import numpy as np
from loggers import logger
from process import input_data, preprocess, modeling

# 1. parser 객체 생성
parser = argparse.ArgumentParser(description='Click & Select')

# 2. 사용할 인수 등록,  이름/타입/help
parser.add_argument('-pth', '--PATH',  type=str, help='Path of Data')
parser.add_argument('-item_id', '--item_id_var', type=str, help='Item ID vairable')
parser.add_argument('-user_id', '--user_id_var', type=str, help='User ID variable')
parser.add_argument('-event', '--event', type=str, help='Event variable')
parser.add_argument('-num', '--num', type=int, help='num variable')

parser.add_argument('-user_var', '--user_var', nargs='+', type=str, help='User variable', default=[])
parser.add_argument('-item_var', '--item_var', nargs='+', type=str, help='Item variable', default=[])
parser.add_argument('-model_type', '--model_type', type=str, help='model type')
args = parser.parse_args()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore

def set_logger(log_name):
    log_obj = logger.AutoMLLog(log_name)
    log_obj.set_handler('automl_process')
    log_obj.set_formats()
    auto_logger = log_obj.addOn()
    
    auto_logger.info('logger 세팅')
        

if __name__ == "__main__":
    seed_everything()
    log_name = 'automl_personalize'
    set_logger(log_name)
    #해당 과정(personalize)에서는 딱히 var_list, num_var, obj_var를 받을 필요는 없으나,
    #기존 코드와의 통일성을 위해 그대로 받을 예정
    data, var_list, num_var, obj_var = input_data.Data_load(args.PATH, log_name).read_data()
    df, user_df, item_df, interaction_df = preprocess.Preprocessing(log_name, data, args.item_id_var, args.user_id_var, args.event, args.user_var, args.item_var).get_per_df()
    mm = modeling.Modeling(log_name, df, user_df, item_df, interaction_df, args.item_id_var, args.user_id_var, args.event, args.num, args.model_type)
    
    
    # 입력 예시
    # python main.py -pth storage/movie_data/interaction_data.csv -item_id ITEM_ID -user_id USER_ID -event EVENT_TYPE -num 10 -user_var  -item_var ITEM_NAME CATEGORY_L1 --model_type auto
    # python main.py -pth storage/shop_data/interaction_data.csv -item_id ITEM_ID -user_id USER_ID -event EVENT_TYPE -num 10 -user_var USER_NAME AGE -item_var ITEM_NAME CATEGORY_L1 --model_type auto
    