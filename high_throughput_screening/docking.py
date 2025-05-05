import pandas as pd
from rdkit import Chem
import random
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


def read_smiles_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['SMILES'].tolist()

def create_molecule_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return None
    return mol

def combine_fragments_at_random_bond(fragment1, fragment2):
    # 创建片段1和片段2的副本，以保持原始结构不变
    mol1 = Chem.Mol(fragment1)
    mol2 = Chem.Mol(fragment2)

    # 随机选择片段1和片段2中的一个原子
    atom1 = random.choice(mol1.GetAtoms())
    atom2 = random.choice(mol2.GetAtoms())

    # 创建可编辑的分子对象
    new_mol = Chem.EditableMol(Chem.CombineMols(mol1, mol2))

    # 添加单键连接
    new_mol.AddBond(atom1.GetIdx(), mol1.GetNumAtoms() + atom2.GetIdx(), order=Chem.BondType.SINGLE)

    # 将编辑分子转换为普通的分子对象
    new_mol = new_mol.GetMol()

    # 确保分子的化学结构是有效的
    try:
        Chem.SanitizeMol(new_mol)
    except Exception as e:
        print(f"Failed to sanitize molecule: {e}")
        return None

    return new_mol

def process_csv_files(file1, file2, output_csv):
    smiles_list1 = read_smiles_from_csv(file1)
    smiles_list2 = read_smiles_from_csv(file2)

    new_molecules = []

    for smiles1 in smiles_list1:
        for smiles2 in smiles_list2:
            mol1 = create_molecule_from_smiles(smiles1)
            mol2 = create_molecule_from_smiles(smiles2)

            if mol1 is None or mol2 is None:
                continue  # 跳过无效的SMILES字符串

            combined_mol = combine_fragments_at_random_bond(mol1, mol2)

            if combined_mol is not None:
                # 获取合并后分子的SMILES字符串
                new_smiles = Chem.MolToSmiles(combined_mol)
                new_molecules.append(new_smiles)

                # 可选：绘制合并后的分子结构并展示
                img = Draw.MolToImage(combined_mol, size=(300, 300))
                plt.imshow(img)
                plt.axis('off')  # 不显示坐标轴
                plt.show()

    # 将结果保存到CSV文件
    df = pd.DataFrame({'new_smiles': new_molecules})
    df.to_csv(output_csv, index=False)

# 设置随机种子，以便结果可以复现
random.seed(9)
# 调用函数，传入CSV文件路径和输出CSV文件路径
docking1_path = 'docking_test1.csv'
docking2_path = 'docking_test2.csv'
output_path= 'docking_test_result.csv'
process_csv_files(docking1_path, docking2_path, output_path)
