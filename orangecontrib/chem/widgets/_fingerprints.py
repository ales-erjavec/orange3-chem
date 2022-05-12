
from typing import Callable, Optional, List

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def RDKFingerprint(mol):
    return Chem.RDKFingerprint(mol)


def GenMACCSKeys(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def GetMorganFingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2),


def GetMorganFingerprint_useFeatures(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)


def fingerprint_from_smiles(
        smiles: str, fingerprint: Callable = Chem.RDKFingerprint
) -> Optional[List[float]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    else:
        if mol:
            fp = fingerprint(mol)
            return list(fp)
    return None
