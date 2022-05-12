from rdkit import Chem


def matches(smi: str, patt: Chem.Mol) -> bool:
    if not smi:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    match = mol.GetSubstructMatch(patt)
    return bool(match)
