import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, QED

# ==============================================================================
# 1. DEFINE CORES & R-GROUPS (Using dummy atoms [*] for clean attachment)
# ==============================================================================
# Attachment points:
# [1*] = C5 or C6 position (Hydrophobic probe for ALA38 vs human SER52)
# [2*] = C3 position (Linker + Warhead for ARG98 vs human LYS112)

CORES = {
    "7-azaindole_C5": "[1*]c1cc2c([2*])c[nH]c2nc1",
    "7-azaindole_C6": "[1*]c1ccc2c([2*])c[nH]c2n1",
    "2-amino-7-azaindole_C5": "[1*]c1cc2c([2*])c(N)[nH]c2nc1",
    "N1-Me-7-azaindole_C5": "[1*]c1cc2c([2*])cn(C)c2nc1",
}

# Vector A: C5/C6 Lipophilic & Steric Probes (Targeting ALA38 / ILE173)
R_C5_C6 = {
    "H": "[1*][H]",
    "F": "[1*]F",
    "Cl": "[1*]Cl",
    "Br": "[1*]Br",
    "Me": "[1*]C",
    "CF3": "[1*]C(F)(F)F",
    "cPr": "[1*]C1CC1",
    "OCHF2": "[1*]OC(F)F",
}

# Vector B (Tier 1: Synthesizable via Amide, Suzuki, Click, Sulfonamide)
R_C3_SYNTH = {
    # Amide couplings (from 7-azaindole-3-acetic acid)
    "amide_acetic_acid": "[2*]CC(=O)NCC(=O)O",
    "amide_cyanomethyl": "[2*]CC(=O)NCC#N",
    "amide_sulfone": "[2*]CC(=O)NCCS(=O)(=O)C",
    "amide_pyridyl": "[2*]CC(=O)Nc1ccncc1",
    "amide_pyrazole": "[2*]CC(=O)Nc1cnn(C)c1",
    "amide_oxetane": "[2*]CC(=O)NC1COC1",
    # Suzuki couplings (from 3-bromo-7-azaindole)
    "suzuki_benzoic_acid": "[2*]c1ccc(C(=O)O)cc1",
    "suzuki_meta_acid": "[2*]c1cccc(C(=O)O)c1",
    "suzuki_benzonitrile": "[2*]c1ccc(C#N)cc1",
    "suzuki_phenylsulfone": "[2*]c1ccc(S(=O)(=O)C)cc1",
    "suzuki_pyridyl_Noxide": "[2*]c1cc[n+]([O-])cc1",
    "suzuki_pyrimidine": "[2*]c1cncnc1",
    # Click chemistry (1,2,3-triazoles from 3-ethynyl-7-azaindole)
    "click_triazole_acid": "[2*]c1cn(CC(=O)O)nn1",
    "click_triazole_amide": "[2*]c1cn(CC(=O)N)nn1",
    "click_triazole_nitrile": "[2*]c1cn(CC#N)nn1",
    "click_triazole_cPr": "[2*]c1cn(C2CC2)nn1",
    # Sulfonamides
    "sulfonamide_Me": "[2*]S(=O)(=O)NC",
    "sulfonamide_acid": "[2*]S(=O)(=O)NCC(=O)O",
}

# Vector B (Tier 2: De Novo Rigidified Linkers & Bioisosteres)
R_C3_DENOVO = {
    # Rigid alkynes to project directly into ARG98 without rotatable bond loss
    "alkyne_acid": "[2*]C#CC(=O)O",
    "alkyne_propargyl_alcohol": "[2*]C#CCO",
    "alkyne_pyridyl": "[2*]C#Cc1ccccn1",
    "alkyne_tetrazole": "[2*]C#Cc1nnn[nH]1",
    # Oxadiazole & Tetrazole bioisosteres
    "oxadiazole_Me": "[2*]c1nc(C)no1",
    "oxadiazole_acid": "[2*]c1nc(CC(=O)O)no1",
    "aryl_tetrazole": "[2*]c1ccc(-c2nnn[nH]2)cc1",
    # Spirocyclic / Constrained H-bond acceptors
    "spiro_oxetane_amide": "[2*]C(=O)N1CC2(C1)COC2",
    "bicyclo_acid": "[2*]C12CCC(C(=O)O)(CC1)CC2",
    "acylsulfonamide": "[2*]c1ccc(C(=O)NS(=O)(=O)C)cc1",
}


# ==============================================================================
# 2. ASSEMBLY & PAINS/PROPERTY FILTERING FUNCTIONS
# ==============================================================================
def attach_groups(core_smiles, r1_smiles, r2_smiles):
    """Combines core with R1 [1*] and R2 [2*] using RDKit reaction/zip."""
    rxn1 = AllChem.ReactionFromSmarts("[*:1]-[1*].[*:2]-[1*]>>[*:1]-[*:2]")
    rxn2 = AllChem.ReactionFromSmarts("[*:1]-[2*].[*:2]-[2*]>>[*:1]-[*:2]")

    mol_core = Chem.MolFromSmiles(core_smiles)
    mol_r1 = Chem.MolFromSmiles(r1_smiles)
    mol_r2 = Chem.MolFromSmiles(r2_smiles)

    # Attach R1
    ps1 = rxn1.RunReactants((mol_core, mol_r1))
    if not ps1:
        return None
    inter = ps1[0][0]
    Chem.SanitizeMol(inter)

    # Attach R2
    ps2 = rxn2.RunReactants((inter, mol_r2))
    if not ps2:
        return None
    prod = ps2[0][0]
    Chem.SanitizeMol(prod)
    return Chem.RemoveHs(prod)


# Initialize PAINS filter
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(params)

# ==============================================================================
# 3. ENUMERATE LIBRARY
# ==============================================================================
all_r2 = {}
for k, v in R_C3_SYNTH.items():
    all_r2[k] = ("Synthesizable", v)
for k, v in R_C3_DENOVO.items():
    all_r2[k] = ("De_Novo", v)

records = []
seen_smiles = set()

for core_name, core_smi in CORES.items():
    for r1_name, r1_smi in R_C5_C6.items():
        for r2_name, (tier, r2_smi) in all_r2.items():
            mol = attach_groups(core_smi, r1_smi, r2_smi)
            if mol is None:
                continue

            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if can_smi in seen_smiles:
                continue
            seen_smiles.add(can_smi)

            # Filter out PAINS
            if pains_catalog.HasMatch(mol):
                continue

            # Calculate physicochemical properties
            mw = Descriptors.MolWt(mol)
            clogp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            qed_score = QED.qed(mol)

            # Drug-like & Parasite-membrane permeability filter
            # Allows slightly higher TPSA (up to 130) to accommodate carboxylic acids/tetrazoles targeting ARG98
            if mw <= 500 and clogp <= 5.0 and tpsa <= 130 and rotb <= 8:
                records.append(
                    {
                        "ID": f"{core_name}__{r1_name}__{r2_name}",
                        "Tier": tier,
                        "Core": core_name,
                        "R1_C5_C6": r1_name,
                        "R2_C3": r2_name,
                        "SMILES": can_smi,
                        "MW": round(mw, 2),
                        "cLogP": round(clogp, 2),
                        "TPSA": round(tpsa, 2),
                        "HBD": hbd,
                        "HBA": hba,
                        "RotBonds": rotb,
                        "QED": round(qed_score, 3),
                    }
                )

df = pd.DataFrame(records)
df = df.sort_values(by=["Tier", "QED"], ascending=[False, False])
df.to_csv("PfHsp90_7Azaindole_Library.csv", index=False)

print(f"Generated {len(df)} unique, PAINS-free ligands.")
print(df["Tier"].value_counts())
print("\nTop 5 Preview:")
print(df[["ID", "Tier", "MW", "cLogP", "TPSA", "QED"]].head())
