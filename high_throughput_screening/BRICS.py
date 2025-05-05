# from rdkit.Chem import Recap
# from rdkit.Chem import AllChem as Chem
#
# m = Chem.MolFromSmiles('COc1c(/C=C(\\C#N)c2ccc(-c3cc[n+](C)cc3)cc2)ccc2cc(N(C)C)ccc12')
# hierarch = Recap.RecapDecompose(m)
#
# # 叶子节点函数：hierarch.GetLeaves()
# print(hierarch.GetLeaves().keys())
#
# # 子孙节点函数：hierarch.GetAllChildren()
# print(hierarch.GetAllChildren().keys())
#
# # 祖先节点函数，返回列表：getUltimateParents()
# print(hierarch.getUltimateParents()[0].smiles)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import re

input_csv = 'AIEgens.csv'
output_csv = 'docking_test1.csv'

df = pd.read_csv(input_csv)

unique_fragments = set()

# 定义用于匹配类似 [14*] 的正则表达式
pattern = r'\[\d+\*\]|\(\[\d+\*\]\)'

for smiles in df['SMILES']:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 使用 BRICS 算法分解分子
        fragments = BRICS.BRICSDecompose(mol)
        for fragment in fragments:
            # 确保 fragment 是有效的 ROMol 对象
            smiles_cleaned = re.sub(pattern, '', fragment)
            unique_fragments.add(smiles_cleaned)

output_df = pd.DataFrame({'Fragment SMILES': list(unique_fragments)})
output_df.to_csv(output_csv, index=False)

print(f"Unique fragments saved to {output_csv}")
