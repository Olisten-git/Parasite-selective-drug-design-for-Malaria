import os
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_selective_parasite_ligands():
    # Base Smiles Dictionary for tracking modifications
    designs = {}
    
    # Starting native ligand (ADP)
    designs['ADP_native'] = "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@@H]1O"
    
    # 1. Modifying Core: exocyclic Amine -> Cl (for ALA38/SER52 handle) + adding extended arm for ASN140 vacancy
    designs['Modification_1_Core_Only'] = "ClC1=NC=NC2=C1N(CC3=CC=C(CN)C=C3)CN2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@@H]1O"
    
    # 2. Rigidifying Linker: Locked Nucleic Acid (LNA) bicyclic core to trap the parasite glycosidic torsion
    # This introduces a methylene bridge between 2'-O and 4'-C
    designs['Modification_2_Locked_Linker'] = "ClC1=NC=NC2=C1N(CC3=CC=C(CN)C=C3)CN2[C@@H]1[C@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H]1OC2"
    
    # 3. Replacing Tail: Strip diphosphate, replace with bulky Naphthalene to fill the open P15-P20 loop cavity
    designs['Modification_3_Ultimate_Selective'] = "ClC1=NC=NC2=C1N(CC3=CC=C(CN)C=C3)CN2[C@@H]1[C@H]2O[C@H](COC(=O)CC4=CC5=CC=CC=C5C=C4)[C@@H]1OC2"

    print("Generating 3D conformations and saving SDF files for docking...")
    
    for name, smiles in designs.items():
        print(f"\nProcessing: {name}")
        # Parse SMILES and add hydrogen atoms essential for proper 3D geometry
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error parsing SMILES for {name}")
            continue
            
        mol_h = Chem.AddHs(mol)
        
        # Generate initial 3D embed coordinate space using ETKDGv3 algorithm
        embed_params = AllChem.ETKDGv3()
        embed_status = AllChem.EmbedMolecule(mol_h, embed_params)
        
        if embed_status != 0:
            print(f"Warning: Initial embedding failed for {name}, trying robust settings...")
            AllChem.EmbedMolecule(mol_h, useRandomCoords=True, maxAttempts=1000)
            
        # Minimize and optimize structure geometry using MMFF94 force field
        try:
            ff_status = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
            print(f"MMFF94 Optimization status for {name}: {'Success' if ff_status == 0 else 'Incomplete convergence'}")
        except Exception as e:
            print(f"MMFF94 optimization failed for {name}, falling back to UFF. Error: {e}")
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            
        # Write out to clean 3D SDF output format for down-stream molecular docking pipelines
        filename = f"{name}_docking_ready.sdf"
        writer = Chem.SDWriter(filename)
        writer.write(mol_h)
        writer.close()
        print(f"Successfully created: {filename}")

if __name__ == "__main__":
    generate_selective_parasite_ligands()
