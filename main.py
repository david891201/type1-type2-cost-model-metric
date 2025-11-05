import warnings

from src.utils import read_csv_files, check_column_consistency, load_selected_feature, load_config
from src.preprocessing.preprocessing import standardscaler_transform, dummies_transform, handle_missing_values

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")
#載入必要的參數設定
configs = load_config()
#讀取訓練集和測試集
print("*****讀取資料集*****")
dfs_dict = read_csv_files(configs)
print("*****確保訓練集和測試集的column數量和順序一致*****")
dfs_dict = check_column_consistency(dfs_dict)
#資料前處理：包含排除特定樣本、填補缺失值、數值型特徵標準化、類別型特徵轉為dummies
print("*****排除特定樣本和填補缺失值*****")
dfs_dict = handle_missing_values(configs, dfs_dict)
print("*****對數值型特徵進行標準化，以及將類別型特徵轉換為dummies*****")
dfs_dict = standardscaler_transform(configs, dfs_dict)
dfs_dict = dummies_transform(configs, dfs_dict)
print("*****確保經過資料前處理後訓練集和測試集，兩者的column數量和順序一致*****")
dfs_dict = check_column_consistency(dfs_dict)
#載入經過特徵篩選留下的特徵清單
print("*****載入經過特徵篩選留下的特徵清單*****")
selected_feature_names_loaded = load_selected_feature(configs.get("selected_feature_path"))
#訓練集和測試集都只留下存在於特徵清單的特徵
print("*****訓練集和測試集都只留下存在於特徵清單的特徵*****")
X_train = dfs_dict["train"][selected_feature_names_loaded]
y_train = dfs_dict["train"]["Y"]
X_test =  dfs_dict["test"][selected_feature_names_loaded]
y_test = dfs_dict["test"]["Y"]

#本次用來測試的門檻值為1%、2%、3%、4%、5%、10%、20%、30%、......、100%
thresholds_05_10 = [0.01, 0.02, 0.03, 0.04, 0.05]
thresholds_10_100 = np.arange(0.1, 1.05, 0.1)
thresholds = np.concatenate([thresholds_05_10, thresholds_10_100])

#每種模型先經過超參數調校，選出一組在該模型表現最好的超參數，再使用表現最好的那組超參數訓練，訓練後各模型和其他模型比較，最終找出套用在測試集成效後成本最低的模型。
#本專案事前已使用StratifiedKfold交叉驗證進行超參數調校，得到各模型於訓練時使用的超參數組合，超參數調教過程使用的評估指標為本次研究提出基於成本的新指標（也就是和評估測試集成效所使用的指標一樣）
rf = RandomForestClassifier(**configs.get("model_params")["random_forest"])
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb = XGBClassifier(**configs.get("model_params")["xgboost"], scale_pos_weight=scale_pos_weight)
voting_clf = VotingClassifier(**configs.get("model_params")["voting_classifier"],estimators=[('xgb', xgb), ('rf', rf)])
model_list = {"rf":rf, "xgb":xgb, "vot":voting_clf}
best_model={"model": None, "threshold": None, "min_cost":np.inf}
output_file = configs.get("output_file_path")

#將每個門檻對應的成本儲存在.xlsx裡面
print("*****開始比較各模型套用在測試集的成效*****")
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for model_name, model in model_list.items(): 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results=[]
        for threshold in thresholds:
            #找出每個門檻下的基準值，例如：門檻1%就是要找機率值大小為前1%(也就是PR99)的樣本，找到該門檻的基準值後，只要大於等於基準值即為符合條件的樣本
            percentage = np.percentile(y_prob, 100-threshold*100) #各門檻的基準值
            selected = y_prob >= percentage  
            num_selected = np.sum(selected)  #統計大於等於基準值的樣本有幾筆
            num_true_positive = np.sum(y_test[selected])
            num_true_positive_percentage = num_true_positive/np.sum(y_test)
            num_times = num_true_positive/(np.sum(y_test)*threshold)
            cost = configs.get("cost_of_inspection")*num_selected + configs.get("cost_of_punishment")*(np.sum(y_test)-num_true_positive)

            results.append([int(threshold * 100), num_selected, num_true_positive, num_true_positive_percentage, num_times, cost])
        
        df = pd.DataFrame(results, columns=["%", ">=門檻的樣本數", "Fraud N", "Capture Fraud N", "Capture Times", "成本"])

        min_cost_row = df.loc[df['成本'].idxmin()]
        min_threshold = min_cost_row["%"]
        min_cost = min_cost_row["成本"]

        if min_cost<best_model["min_cost"]:
            best_model["model"]=model_name
            best_model["min_cost"]=min_cost
            best_model["threshold"]=min_threshold

        df.to_excel(writer, sheet_name=f"{model_name}", index=False)
print("*****最終表現最好的模型*****")
print(best_model)