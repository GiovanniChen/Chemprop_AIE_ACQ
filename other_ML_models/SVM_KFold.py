import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# 读取包含SMILES和标签的数据
file_path = "C:/BIT/codes/chemprop/data/AIE_ACQ/AIEACQ_train.csv"
column_names = ["SMILES", "Label"]
data = pd.read_csv(file_path, usecols=column_names)

# 使用LabelEncoder将目标标签编码为整数
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# 通过RDKit计算Morgan指纹作为特征
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# 创建预处理和模型训练的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('svm', SVC(probability=True))  # SVM分类器
])

# 设置超参数网格
param_grid = {
    'svm__kernel': ['linear', 'rbf'],  # 核函数类型
    'svm__C': [0.1, 1, 10],  # 正则化参数
    'svm__gamma': ['scale', 'auto']  # RBF核函数的参数，它定义了单个训练样本的影响范围，可以看作是核函数的“带宽”。
    # gamma值较小意味着更远的影响范围，而较大的gamma值意味着更近的影响范围
}

# 初始化KFold进行五折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化GridSearchCV，使用五折交叉验证
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kf, scoring='roc_auc_ovr', n_jobs=-1, verbose=2)
#f1_weighted

# 计算特征
features = features.values
labels = data["Label"].values

# 执行超参数优化
grid_search.fit(features, labels)

# 打印最佳超参数组合
print("Best parameters found:")
print(grid_search.best_params_)
#
# # 保存最佳模型和特征缩放器
# model_filename = "20240424_svm_model_GridSearchCV_best.pkl"
# scaler_filename = "20240424_svm_scaler_GridSearchCV_best.pkl"
#
# joblib.dump(grid_search.best_estimator_, model_filename)
# joblib.dump(grid_search.best_estimator_.named_steps['scaler'], scaler_filename)
#
# print(f"Best model and scaler saved to {model_filename} and {scaler_filename}")