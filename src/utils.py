import os

import pandas as pd
import pickle
import yaml

def read_csv_files(configs: dict):
    dfs_dict = {}
    for file in os.listdir(configs.get("data_path")): #data_path: 放置訓練集和測試集檔案的資料夾路徑
        if file.endswith('.csv'):
            file_path = os.path.join(configs.get("data_path"), file)
            df = pd.read_csv(file_path, encoding='big5')
            # dfs_dict這個字典會有兩個key，一個為train另一個為test，key下各自對應訓練集和測試集
            dfs_dict[os.path.splitext(file)[0]] = df 
    return dfs_dict

def check_column_consistency(dfs_dict: dict, reference_key: str = "train"): #reference_key:以訓練集存在的column以及column順序為標準
    """
    確保訓練集和測試集的column數量和順序一致
    """
    if not dfs_dict:
        print("沒有讀取到任何 DataFrame ！")
        return False
    reference_columns = dfs_dict.get(reference_key).columns.tolist()
    for key, df in dfs_dict.items():
        if key != reference_key:
            columns_difference = len(df.columns.to_list()) - len(reference_columns)
            if df.columns.tolist() != reference_columns:
                print(f"{key} data 的欄位順序或數量與 {reference_key} data 不同！ 欄位數量比預期多出了 {columns_difference} 列")
                try:
                    dfs_dict[key] = df[reference_columns]
                except Exception as e:
                    print(e) #因為類別型特徵轉換成dummies，有可能發生訓練集有的column，但在測試集找不到
                    missing_columns = list(set(reference_columns) - set(df.columns))
                    for missing_col in missing_columns:
                        df[missing_col] = 0 #因為確定這些column都是類別型的column，因此可以直接補0
                    dfs_dict[key] = df[reference_columns] #確認測試集column的順序和訓練集一致
                print("經調整後測試集和訓練集的欄位順序與數量為一致！")
            else:
                print("測試集和訓練集的欄位順序與數量為一致，無須調整！")
        else:
            pass
    return dfs_dict

def load_selected_feature(feature_list):
    with open(feature_list, 'rb') as f:
        selected_feature_names_loaded = pickle.load(f)
    return selected_feature_names_loaded

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
