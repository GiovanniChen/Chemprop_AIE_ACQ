import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef

# 加载保存的MLP模型
model_file = "mlp_model.pkl" 
model = joblib.load(model_file)

# 读取CSV文件
file_path = "AIEACQ_test.csv"
data = pd.read_csv(file_path)

# 假设CSV文件中SMILES数据在名为'SMILES'的列中
smiles_list = data['SMILES'].tolist()

# 对SMILES数据进行预处理，这里假设你之前已经计算过Morgan指纹
# 如果需要重新计算，请取消下面注释并进行相应的处理
data['Morgan_Features'] = data['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

# 将Morgan指纹转换为DataFrame
new_features = pd.DataFrame(data['Morgan_Features'].apply(lambda x: pd.Series(list(x.ToBitString()))), dtype=int)

# # 使用加载的模型进行预测
# predictions = model.predict(new_features)

# 使用模型进行预测，获取概率
probabilities = model.predict_proba(new_features)

# 获取预测的类别标签
predicted_labels = probabilities.argmax(axis=1)

# 获取真实标签
true_labels = data['Label']

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
predictions_df = pd.DataFrame({'SMILES': data['SMILES'], 'Label': predicted_labels})

# 将预测结果保存到新的CSV文件中
output_data = pd.DataFrame({'SMILES': smiles_list, 'Prediction': predicted_labels})
output_file_path = "MLP_pred.csv"
output_data.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
