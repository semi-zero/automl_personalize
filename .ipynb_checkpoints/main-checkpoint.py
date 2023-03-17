import argparse
import logging
import pandas as pd
import random
import os
import numpy as np
from loggers import logger
from process import input_data#, modeling, preprocess

# 1. parser 객체 생성
parser = argparse.ArgumentParser(description='Click & Select')

# 2. 사용할 인수 등록,  이름/타입/help
parser.add_argument('-pth', '--PATH',  type=str, help='Path of Data')
parser.add_argument('-item_id', '--item_id_var', type=str, help='Item ID vairable')
parser.add_argument('-user_id', '--user_id_var', type=str, help='User ID variable')
parser.add_argument('-event', '--event', type=str, help='Event variable')


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
    data, var_list, num_var, obj_var = input_data.Data_load(args.PATH, log_name).read_data()
    #df = preprocess.Preprocessing(log_name, data, var_list, num_var, obj_var, args.target, args.unique_id, ).get_df()
    #mm = modeling.Modeling(log_name, df, obj_var = obj_var, target=args.target, unique_id = args.unique_id, model_type=args.model_type, OVER_SAMPLING=args.OVER_SAMPLING, HPO=args.HPO)
    
    
    # 입력 예시
    # python main.py -pth storage/movie_data/interaction_data.csv -item_id ITEM_ID -user_id USER_ID -event EVENT_TYPE
    