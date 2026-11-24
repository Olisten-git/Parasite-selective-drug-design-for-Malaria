#!/usr/bin/env python3
"""
7-azaindole analogue generator (RDKit)

Generates a small, traceable library of single- and selected double-substituted
7-azaindole analogues from the user's supplied 1H-pyrrolo[2,3-b]pyridine
structure. It writes:
  output_azaindole/azaindole_library.sdf
  output_azaindole/azaindole_library.csv
  output_azaindole/substitution_sites.txt

Requirements:
  pip install rdkit

Run:
  python generate_azaindole_library.py

Notes:
- Substitution sites are identified computationally as aromatic carbon atoms
  carrying replaceable hydrogen; no assumed C-numbering is imposed.
- Atom labels in the output refer to 1-based atom indices in the heavy-atom
  parent molecule after RDKit parses the supplied mol block.
- 3D conformers are generated for ligand preparation, not protein-bound poses.
- MMFF94s is used where parameterised; UFF is the fallback.
"""

from pathlib import Path
from itertools import combinations
import csv
import math
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, QED

# ---------------------------------------------------------------------------
# 1. Parent structure: mol block supplied by the user
# ---------------------------------------------------------------------------

PARENT_MOLBLOCK = r"""
9222
  -OEChem-10292605592D

 15 16  0     0  0  0  0  0  0999 V2000
    4.6783   -0.6077    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660   -0.8030    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.7321    0.6970    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7321   -0.3030    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.6783    1.0018    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660    1.1970    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2619    0.1970    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.6970    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -0.3030    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.8709    1.5911    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.8709   -1.1970    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.8660    1.8170    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.8819    0.1970    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4631    1.0070    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4631   -0.6130    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  4  1  0  0  0  0
  1  7  1  0  0  0  0
  1 11  1  0  0  0  0
  2  4  1  0  0  0  0
  2  9  2  0  0  0  0
  3  4  2  0  0  0  0
  3  5  1  0  0  0  0
  3  6  1  0  0  0  0
  5  7  2  0  0  0  0
  5 10  1  0  0  0  0
  6  8  2  0  0  0  0
  6 12  1  0  0  0  0
  7 13  1  0  0  0  0
  8  9  1  0  0  0  0
  8 14  1  0  0  0  0
  9 15  1  0  0  0  0
M  END
"""

# Substituent SMILES use [*:1] as the attachment atom.
# The first atom bonded to [*:1] is connected to the selected parent atom.
SUBSTITUENTS = {
    "methyl":       "[*:1]C",
    "ethyl":        "[*:1]CC",
    "isopropyl":    "[*:1]C(C)C",
    "cyclopropyl":  "[*:1]C1CC1",
    "fluoro":       "[*:1]F",
    "chloro":       "[*:1]Cl",
    "cyano":        "[*:1]C#N",
    "carboxamide":  "[*:1]C(=O)N",
    "hydroxymethyl":"[*:1]CO",
}

# Curated double-substitution combinations. Each pair is applied to every
# distinct pair of eligible carbon sites. This is intentionally limited.
DOUBLE_COMBINATIONS = [
    ("methyl", "fluoro"),
    ("methyl", "cyano"),
    ("cyclopropyl", "fluoro"),
    ("cyclopropyl", "carboxamide"),
]

# Set True to include N-alkyl analogues that replace the pyrrolic [nH].
# These are kept separate because N-alkylation removes the parent NH donor.
INCLUDE_N_ALKYL = True
N_SUBSTITUENTS = {
    "N_methyl": "[*:1]C",
    "N_ethyl": "[*:1]CC",
}

OUTPUT_DIR = Path("output_azaindole")
N_CONFORMERS = 8
RANDOM_SEED = 20261118
MAX_DOUBLE_PRODUCTS = 2000


# ---------------------------------------------------------------------------
# 2. Parsing and site identification
# ---------------------------------------------------------------------------

def load_parent():
    mol = Chem.MolFromMolBlock(
        PARENT_MOLBLOCK.strip(),
        removeHs=True,
        sanitize=True,
        strictParsing=True,
    )
    if mol is None:
        raise ValueError("RDKit could not parse the supplied 7-azaindole mol block.")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def find_carbon_substitution_sites(parent):
    """Return aromatic carbon atoms bearing at least one replaceable H."""
    sites = []
    for atom in parent.GetAtoms():
        if (
            atom.GetAtomicNum() == 6
            and atom.GetIsAromatic()
            and atom.GetTotalNumHs() > 0
        ):
            sites.append(atom.GetIdx())
    return sites


def find_pyrrolic_nh_sites(parent):
    """Find aromatic [nH] atoms; used only for explicitly flagged N-alkyl analogues."""
    return [
        atom.GetIdx()
        for atom in parent.GetAtoms()
        if atom.GetAtomicNum() == 7
        and atom.GetIsAromatic()
        and atom.GetTotalNumHs() > 0
    ]


# ---------------------------------------------------------------------------
# 3. Attach a substituent to a selected parent atom
# ---------------------------------------------------------------------------

def attach_fragment(parent, target_idx, fragment_smiles):
    """
    Attach a fragment represented as [*:1]R to target_idx.

    The dummy atom is removed after connecting its neighbouring atom to the
    selected parent atom. Sanitisation checks valence and aromaticity.
    """
    parent = Chem.Mol(parent)
    target_atom = parent.GetAtomWithIdx(int(target_idx))
    # For pyrrolic [nH] substitution, explicitly remove the implicit N-H
    # before adding the new N-C bond. Carbon substitutions are unchanged.
    if target_atom.GetAtomicNum() == 7 and target_atom.GetIsAromatic() and target_atom.GetTotalNumHs() > 0:
        target_atom.SetNoImplicit(True)
        target_atom.SetNumExplicitHs(0)

    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:
        raise ValueError(f"Could not parse substituent SMILES: {fragment_smiles}")

    dummy_atoms = [
        atom for atom in fragment.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == 1
    ]
    if len(dummy_atoms) != 1:
        raise ValueError(f"Substituent must contain exactly one [*:1]: {fragment_smiles}")

    dummy = dummy_atoms[0]
    if dummy.GetDegree() != 1:
        raise ValueError(f"Attachment dummy must have one neighbour: {fragment_smiles}")

    fragment_neighbour_idx = dummy.GetNeighbors()[0].GetIdx()
    combined = Chem.CombineMols(parent, fragment)
    rw = Chem.RWMol(combined)

    fragment_offset = parent.GetNumAtoms()
    dummy_idx = fragment_offset + dummy.GetIdx()
    neighbour_idx = fragment_offset + fragment_neighbour_idx

    rw.AddBond(int(target_idx), int(neighbour_idx), Chem.BondType.SINGLE)
    rw.RemoveAtom(int(dummy_idx))

    product = rw.GetMol()
    Chem.SanitizeMol(product)
    product = Chem.RemoveHs(product)
    Chem.AssignStereochemistry(product, cleanIt=True, force=True)
    return product


def canonical_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


# ---------------------------------------------------------------------------
# 4. Generate analogue records
# ---------------------------------------------------------------------------

def generate_library(parent, carbon_sites):
    records = []
    seen = set()

    def add_record(mol, name, family, sites, substituent_names, note=""):
        smi = canonical_smiles(mol)
        if smi in seen:
            return
        seen.add(smi)
        records.append({
            "mol": mol,
            "compound_id": f"AZA{len(records)+1:04d}",
            "name": name,
            "family": family,
            "site_atom_indices_1based": ";".join(str(i + 1) for i in sites),
            "substituents": ";".join(substituent_names),
            "canonical_smiles": smi,
            "design_note": note,
        })

    add_record(
        parent,
        "7-azaindole_parent",
        "parent",
        [],
        [],
        "Unmodified reference scaffold",
    )

    # Single carbon substitutions at every eligible aromatic carbon.
    for site in carbon_sites:
        for sub_name, sub_smiles in SUBSTITUENTS.items():
            try:
                product = attach_fragment(parent, site, sub_smiles)
                add_record(
                    product,
                    f"7-azaindole_{sub_name}_at_atom{site+1}",
                    "single_substitution",
                    [site],
                    [sub_name],
                    "Position is an RDKit heavy-atom index, not an asserted IUPAC locant",
                )
            except Exception as exc:
                print(
                    f"[WARNING] Skipping {sub_name} at atom {site+1}: {exc}",
                    file=sys.stderr,
                )

    # Selected double substitutions at distinct eligible carbon atoms.
    generated_double = 0
    for sub1_name, sub2_name in DOUBLE_COMBINATIONS:
        for site1, site2 in combinations(carbon_sites, 2):
            try:
                first = attach_fragment(parent, site1, SUBSTITUENTS[sub1_name])
                # The parent atom indices are unchanged by the first attachment:
                # the fragment is appended after the parent atoms.
                product = attach_fragment(first, site2, SUBSTITUENTS[sub2_name])
                add_record(
                    product,
                    f"7-azaindole_{sub1_name}_atom{site1+1}_{sub2_name}_atom{site2+1}",
                    "double_substitution",
                    [site1, site2],
                    [sub1_name, sub2_name],
                    "Two substitutions at distinct aromatic carbon sites",
                )
                generated_double += 1
                if generated_double >= MAX_DOUBLE_PRODUCTS:
                    break
            except Exception as exc:
                print(
                    f"[WARNING] Skipping double substitution "
                    f"{sub1_name}/{sub2_name} at atoms {site1+1}/{site2+1}: {exc}",
                    file=sys.stderr,
                )
        if generated_double >= MAX_DOUBLE_PRODUCTS:
            break

    # Optional N-alkyl series: explicitly separated because it removes the NH donor.
    if INCLUDE_N_ALKYL:
        n_sites = find_pyrrolic_nh_sites(parent)
        for n_site in n_sites:
            for sub_name, sub_smiles in N_SUBSTITUENTS.items():
                try:
                    product = attach_fragment(parent, n_site, sub_smiles)
                    add_record(
                        product,
                        f"7-azaindole_{sub_name}_at_atom{n_site+1}",
                        "N_substitution_NH_donor_removed",
                        [n_site],
                        [sub_name],
                        "Exploratory N-alkyl analogue; parent pyrrolic NH donor is removed",
                    )
                except Exception as exc:
                    print(
                        f"[WARNING] Skipping {sub_name} at N atom {n_site+1}: {exc}",
                        file=sys.stderr,
                    )

    return records


# ---------------------------------------------------------------------------
# 5. Descriptors and simple triage flags
# ---------------------------------------------------------------------------

def calculate_descriptors(mol):
    mw = Descriptors.MolWt(mol)
    clogp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    qed = QED.qed(mol)

    # Broad exploratory window only; not a pass/fail drug-likeness verdict.
    descriptor_window = (
        180 <= mw <= 400
        and 0 <= clogp <= 4
        and 0 <= hbd <= 3
        and 1 <= hba <= 7
        and 30 <= tpsa <= 120
        and rotb <= 5
    )

    return {
        "MW_Da": round(mw, 2),
        "cLogP_RDKit": round(clogp, 2),
        "TPSA_A2": round(tpsa, 2),
        "HBD": int(hbd),
        "HBA": int(hba),
        "RotatableBonds": int(rotb),
        "RingCount": int(rings),
        "HeavyAtomCount": int(heavy),
        "QED": round(qed, 3),
        "within_broad_descriptor_window": bool(descriptor_window),
    }


# ---------------------------------------------------------------------------
# 6. Generate one low-energy 3D conformer per molecule
# ---------------------------------------------------------------------------

def make_low_energy_3d(mol, n_conformers=N_CONFORMERS):
    mol_h = Chem.AddHs(Chem.Mol(mol))

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.numThreads = 0
    conf_ids = list(AllChem.EmbedMultipleConfs(
        mol_h,
        numConfs=n_conformers,
        params=params,
    ))

    if not conf_ids:
        raise RuntimeError("ETKDG failed to generate a conformer.")

    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol_h,
            numThreads=0,
            maxIters=1000,
            mmffVariant="MMFF94s",
        )
        method = "MMFF94s"
    elif AllChem.UFFHasAllMoleculeParams(mol_h):
        results = AllChem.UFFOptimizeMoleculeConfs(
            mol_h,
            numThreads=0,
            maxIters=1000,
        )
        method = "UFF"
    else:
        raise RuntimeError("Neither MMFF94s nor UFF parameters are available.")

    # Each result is (convergence_status, energy).
    best_position = min(range(len(results)), key=lambda i: results[i][1])
    best_conf_id = conf_ids[best_position]
    best_energy = float(results[best_position][1])

    best_conf = Chem.Conformer(mol_h.GetConformer(best_conf_id))
    output = Chem.Mol(mol_h)
    output.RemoveAllConformers()
    output.AddConformer(best_conf, assignId=True)

    return output, method, best_energy


# ---------------------------------------------------------------------------
# 7. Main workflow and file export
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parent = load_parent()
    carbon_sites = find_carbon_substitution_sites(parent)
    nh_sites = find_pyrrolic_nh_sites(parent)

    if not carbon_sites:
        raise RuntimeError(
            "No aromatic carbon atoms bearing hydrogen were identified. "
            "Check the input mol block and RDKit aromaticity perception."
        )

    print("\nParent scaffold")
    print("---------------")
    print("Canonical SMILES:", canonical_smiles(parent))
    print("Heavy atoms:", parent.GetNumAtoms())
    print("Aromatic carbon substitution sites (1-based atom indices):",
          [i + 1 for i in carbon_sites])
    print("Pyrrolic NH sites (1-based atom indices):",
          [i + 1 for i in nh_sites])

    # Write a human-readable atom/site map for interpreting the library.
    site_file = OUTPUT_DIR / "substitution_sites.txt"
    with site_file.open("w", encoding="utf-8") as handle:
        handle.write("7-AZAINDOLE SUBSTITUTION-SITE MAP\n")
        handle.write("================================\n")
        handle.write("Atom indices are 1-based RDKit heavy-atom indices.\n")
        handle.write("They are not asserted IUPAC locants.\n\n")
        for atom in parent.GetAtoms():
            neighbours = [
                f"{n.GetIdx()+1}:{n.GetSymbol()}"
                for n in atom.GetNeighbors()
            ]
            handle.write(
                f"Atom {atom.GetIdx()+1:2d} | {atom.GetSymbol():2s} | "
                f"aromatic={atom.GetIsAromatic()} | "
                f"H={atom.GetTotalNumHs()} | "
                f"neighbours={', '.join(neighbours)}\n"
            )
        handle.write("\nEligible aromatic carbon sites: ")
        handle.write(", ".join(str(i + 1) for i in carbon_sites))
        handle.write("\nPyrrolic NH sites: ")
        handle.write(", ".join(str(i + 1) for i in nh_sites))
        handle.write("\n")

    records = generate_library(parent, carbon_sites)
    print(f"\nUnique valid molecules generated: {len(records)}")

    sdf_path = OUTPUT_DIR / "azaindole_library.sdf"
    csv_path = OUTPUT_DIR / "azaindole_library.csv"

    csv_rows = []
    writer = Chem.SDWriter(str(sdf_path))
    writer.SetKekulize(True)

    successful = 0
    failed_3d = 0

    for record in records:
        mol = record["mol"]
        desc = calculate_descriptors(mol)

        try:
            mol_3d, method, energy = make_low_energy_3d(mol)
            mol_3d.SetProp("_Name", record["compound_id"] + "_" + record["name"])
            mol_3d.SetProp("Compound_ID", record["compound_id"])
            mol_3d.SetProp("Design_Family", record["family"])
            mol_3d.SetProp("Substitution_Atom_Indices_1based",
                           record["site_atom_indices_1based"])
            mol_3d.SetProp("Substituents", record["substituents"])
            mol_3d.SetProp("Canonical_SMILES", record["canonical_smiles"])
            mol_3d.SetProp("3D_Minimisation_Method", method)
            mol_3d.SetProp("3D_Minimised_Energy_kcal_mol", f"{energy:.4f}")
            mol_3d.SetProp("Design_Note", record["design_note"])
            for key, value in desc.items():
                mol_3d.SetProp(key, str(value))
            writer.write(mol_3d)
            successful += 1
        except Exception as exc:
            failed_3d += 1
            method = "FAILED"
            energy = math.nan
            print(
                f"[WARNING] 3D generation failed for {record['compound_id']} "
                f"({record['name']}): {exc}",
                file=sys.stderr,
            )

        row = {
            "Compound_ID": record["compound_id"],
            "Name": record["name"],
            "Design_Family": record["family"],
            "Substitution_Atom_Indices_1based": record["site_atom_indices_1based"],
            "Substituents": record["substituents"],
            "Canonical_SMILES": record["canonical_smiles"],
            "Design_Note": record["design_note"],
            **desc,
            "3D_Minimisation_Method": method,
            "3D_Minimised_Energy_kcal_mol": (
                round(energy, 4) if math.isfinite(energy) else ""
            ),
        }
        csv_rows.append(row)

    writer.close()

    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            fieldnames = list(csv_rows[0].keys())
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(csv_rows)

    print("\nFinished")
    print("--------")
    print("Valid unique molecules:", len(records))
    print("3D structures written:", successful)
    print("3D generation failures:", failed_3d)
    print("SDF:", sdf_path.resolve())
    print("CSV:", csv_path.resolve())
    print("Site map:", site_file.resolve())
    print("\nImportant: inspect the generated structures before docking. "
          "Descriptor-window flags are triage aids, not evidence of selectivity "
          "or synthesizability.")


if __name__ == "__main__":
    main()
