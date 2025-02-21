import ase


def compute_residues(atoms: ase.Atoms, residues: dict[str, str]) -> dict[str, list[int]]:
    """Find the smiles in the ase.Atoms
    
    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to search for residues.
    residues : dict[str, str]
        A dictionary of residue names and SMILES strings.
    """
    ...
