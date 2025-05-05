import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef

# 加载模型和特征缩放器
model_filename = "svm_model.pkl"
scaler_filename = "svm_scaler.pkl"
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# 读取新的CSV文件
new_data_file_path = "AIEACQ_test.csv"
new_data = pd.read_csv(new_data_file_path)

# 对新数据的SMILES使用RDKit计算Morgan指纹作为特征
new_data['Morgan_Features'] = new_data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
new_features = pd.DataFrame(new_data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# # 使用特征缩放器对新数据进行标准化
# new_features_scaled = scaler.transform(new_features)

# # 使用加载的模型进行预测
# predictions = model.predict(new_features_scaled)

# 使用模型进行预测，获取概率
probabilities = model.predict_proba(new_features)

# 获取预测的类别标签
predicted_labels = probabilities.argmax(axis=1)

# 获取真实标签
true_labels = new_data['Label']

# 计算AUROC
roc_auc = roc_auc_score(true_labels, probabilities[:, 1])  # 选择正类的概率

# 计算AUPRC
average_precision = average_precision_score(true_labels, probabilities[:, 1])

# 计算准确率
accuracy = accuracy_score(true_labels, predicted_labels)

# 使用二元平均F1分数
f1 = f1_score(true_labels, predicted_labels)

# 计算MCC指数
mcc = matthews_corrcoef(true_labels, predicted_labels)

# 打印性能指标
print(f"AUROC: {roc_auc}")
print(f"Accuracy: {accuracy}")
print(f"AUPRC: {average_precision}")
print(f"F1 Score: {f1}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

# 创建一个新的DataFrame来保存预测结果
predictions_df = pd.DataFrame({'SMILES': new_data['SMILES'], 'Label': predicted_labels})

# 将预测结果保存到新的CSV文件中
output_csv_path = "SVM_pred.csv"
predictions_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
