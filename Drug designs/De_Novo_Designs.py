import os
import csv
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

OUTPUT_DIR = os.path.join(os.getcwd(), "Parasite_3K60_Docking_Library")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SDF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "3K60_DeNovo_Library_3D.sdf")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "3K60_DeNovo_Library_Metadata.csv")

CORES = {
    "E_7Aza_5Cl_N1": "Clc1cnc2[nH0]([*:1])ccc2c1",
    "E_7Aza_5Me_N1": "Cc1cnc2[nH0]([*:1])ccc2c1",
    "E_7Aza_5cPr_N1": "C1CC1c1cnc2[nH0]([*:1])ccc2c1",
    "E_7Aza_6Me_N1": "Cc1ccc2cc[nH0]([*:1])c2n1",
    "E_7Aza_5Cl_C3": "Clc1cnc2[nH]cc([*:1])c2c1",
    "E_7Aza_5cPr_C3": "C1CC1c1cnc2[nH]cc([*:1])c2c1",
    "A_7Deaza_7Cl": "Nc1ncnc2[nH0]([*:1])cc(Cl)c12",
    "B_Allopurinol": "O=c1[nH]cnc2[nH0]([*:1])ncc12",
    "B_Mercaptopurine": "S=c1[nH]cnc2[nH0]([*:1])cnc12",
    "B_4Me_Benzimid": "Cc1cccc2[nH0]([*:1])cnc12",
    "A_4Me_Indazole": "Cc1cccc2[nH0]([*:1])ncc12",
    "C_Benzothiazole": "Cc1cccc2nc([*:1])sc12",
    "B_2AminoPyrim": "Nc1nc([*:1])cc(C2CC2)n1",
    "B_Cinnoline": "c1ccc2c(c1)cc([*:1])nn2",
}

LINKERS = {
    "L_cis_Cyclopent": "[*:1][C@H]1CC[C@H]([*:2])C1",
    "L_trans_Cycloprop": "[*:1][C@H]1C[C@H]1[*:2]",
    "L_Piperazine": "[*:1]N1CCN([*:2])CC1",
    "L_Pyrrolidine": "[*:1][C@H]1CCN([*:2])C1",
    "L_123_Triazole": "[*:1]c1cn([*:2])nn1",
    "L_Spiro33": "[*:1]C1(CC2(C1)CN([*:2])C2)c1ccc(C(C)(C)C)cc1",
    "L_Adamantane": "[*:1]C12CC3CC(CC(C3)C1)C2[*:2]",
    "L_Norbornane": "[*:1]C1CC2CCC1C2[*:2]",
    "L_Morpholine": "[*:1]C1CN([*:2])CCO1",
}

WARHEADS = {
    "W_Tavaborole": "[*:2]c1ccc2c(c1)COB2O",
    "W_Deferiprone": "[*:2]n1ccc(=O)c(O)c1C",
    "W_Tetrazole": "[*:2]c1nnn[nH]1",
    "W_AcylSulfonam": "[*:2]C(=O)NS(=O)(=O)C",
    "W_Mesalazine": "[*:2]c1ccc(O)c(C(=O)O)c1",
    "W_BenzoicAcid": "[*:2]c1ccc(C(=O)O)cc1",
    "W_Resorcinol": "[*:2]c1cc(O)cc(O)c1",
    "W_Isoniazid": "[*:2]c1ccncc1C(=O)NN",
    "W_FosfomycinMimic": "[*:2]C1OC1P(=O)(O)O",
}

SERIES_D_FRAGMENTS = {
    "D1_4Me_Benzimidazole": "Cc1cccc2[nH]cnc12",
    "D2_2Amino_4cPr_Pyrim": "Nc1nc(C2CC2)ccn1",
    "D3_Tavaborole_Frag": "OB1OCc2cc(F)ccc21",
    "D4_Deferiprone_Frag": "Cn1ccc(=O)c(O)c1C",
    "D5_Tranylcypromine_Acid": "O=C(O)[C@H]1C[C@H]1c1ccccc1",
    "D6_Pyridyl_Piperazine": "c1ccncc1N1CCNCC1",
    "E7_7Aza_5cPr_6Me": "Cc1nc2[nH]ccc2cc1C1CC1",
    "E7b_7Aza_5CF3": "FC(F)(F)c1cnc2[nH]ccc2c1",
}


def connect_fragments(smiles1, smiles2, map_num):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None

    combo = Chem.CombineMols(mol1, mol2)
    rw = Chem.RWMol(combo)

    dummies = [
        atom for atom in rw.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_num
    ]
    if len(dummies) != 2:
        return None

    d1, d2 = dummies[0], dummies[1]
    n1 = d1.GetNeighbors()[0]
    n2 = d2.GetNeighbors()[0]
    bond_type = d1.GetBonds()[0].GetBondType()

    rw.AddBond(n1.GetIdx(), n2.GetIdx(), bond_type)
    for idx in sorted([d1.GetIdx(), d2.GetIdx()], reverse=True):
        rw.RemoveAtom(idx)

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def passes_druglikeness(mol, is_fragment=False):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if is_fragment:
        return mw <= 300 and logp <= 4.0
    return (220 <= mw <= 550) and (-1.5 <= logp <= 5.5) and (rot_bonds <= 8)


def generate_3d_conformer(mol):
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        if AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass
    return mol_h


def build_library():
    unique_smiles = {}
    records = []

    print("Enumerating combinatorial library across Series A, B, C, D, and E...")

    for frag_name, smi in SERIES_D_FRAGMENTS.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if can_smi not in unique_smiles:
                unique_smiles[can_smi] = (frag_name, "Series_D_E_Fragment", mol)

    for (c_name, c_smi), (l_name, l_smi), (w_name, w_smi) in itertools.product(
        CORES.items(), LINKERS.items(), WARHEADS.items()
    ):
        if l_name in ["L_Spiro33", "L_Adamantane"] and w_name in [
            "W_Tavaborole", "W_Deferiprone", "W_Mesalazine"
        ]:
            continue

        core_linker_smi = connect_fragments(c_smi, l_smi, map_num=1)
        if not core_linker_smi:
            continue

        full_smi = connect_fragments(core_linker_smi, w_smi, map_num=2)
        if not full_smi:
            continue

        mol = Chem.MolFromSmiles(full_smi)
        if mol and passes_druglikeness(mol, is_fragment=False):
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if can_smi not in unique_smiles:
                lig_id = f"{c_name}__{l_name}__{w_name}"
                series_tag = (
                    "Series_E_7Azaindole"
                    if c_name.startswith("E_")
                    else "Series_ABC_DeNovo"
                )
                unique_smiles[can_smi] = (lig_id, series_tag, mol)

    print(f"Total valid unique structures passing drug-likeness filters: {len(unique_smiles)}")
    print("Generating 3D conformers and minimizing energies (this may take 1-2 minutes)...")

    writer = Chem.SDWriter(SDF_OUTPUT_PATH)
    success_count = 0

    for can_smi, (lig_id, series_tag, mol) in unique_smiles.items():
        mol_3d = generate_3d_conformer(mol)
        if mol_3d is None:
            continue

        mw = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

        mol_3d.SetProp("_Name", lig_id)
        mol_3d.SetProp("Series", series_tag)
        mol_3d.SetProp("SMILES", can_smi)
        mol_3d.SetProp("MW", str(mw))
        mol_3d.SetProp("cLogP", str(logp))
        writer.write(mol_3d)

        records.append({
            "Ligand_ID": lig_id,
            "Series": series_tag,
            "SMILES": can_smi,
            "MW": mw,
            "cLogP": logp,
            "HBD": hbd,
            "HBA": hba,
            "TPSA": tpsa,
            "RotBonds": rot,
        })
        success_count += 1
        if success_count % 50 == 0:
            print(f"  Processed {success_count} 3D ligands...")

    writer.close()

    if records:
        fieldnames = [
            "Ligand_ID", "Series", "SMILES", "MW", "cLogP", "HBD", "HBA", "TPSA", "RotBonds"
        ]
        with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
            dict_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(records)

    print("\n========================================================")
    print(f"DONE! Successfully generated {success_count} 3D ligands.")
    print(f"PyRx SDF file saved to : {SDF_OUTPUT_PATH}")
    print(f"Metadata CSV saved to  : {CSV_OUTPUT_PATH}")
    print("========================================================")


if __name__ == "__main__":
    build_library()
