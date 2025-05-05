from rdkit import Chem
from rdkit.Chem import RDConfig
import pandas as pd
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# 读取原始CSV文件路径和输出CSV文件路径
input_csv = 'E:/BIT/codes/chemprop/data/AIE_ACQ/design/test1108.csv'
output_csv = 'E:/BIT/codes/chemprop/data/AIE_ACQ/design/test1108_SAScore.csv'

# 读取CSV文件
df = pd.read_csv(input_csv)

# 提取SMILES列
smiles_list = df['SMILES'].tolist()

# 计算每个SMILES对应的SAscore
scores = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        scores.append(None)  # 处理无法解析的SMILES
    else:
        score = sascorer.calculateScore(mol)
        scores.append(score)

# 将SAscore添加到DataFrame中
df['SAscore'] = scores

# 将结果保存到新的CSV文件
df.to_csv(output_csv, index=False)