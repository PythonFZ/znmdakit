from pathlib import Path

import ase
import ase.atoms
import MDAnalysis as mda
import znh5md
from ase.neighborlist import NeighborList, natural_cutoffs
from MDAnalysis.coordinates.H5MD import H5MDReader
from tqdm import tqdm


def get_bonds(atoms: ase.Atoms) -> list[tuple[int, int]]:
    """Calculate the bonds in an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object for which to calculate bonds.

    Returns
    -------
    list of tuple of int
        A list of tuples, each containing two indices of bonded atoms.
    """
    # Step 1: Calculate natural cutoffs
    cutoffs = natural_cutoffs(atoms)

    # Step 2: Create a neighbor list
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Step 3: Generate the bonds list
    bonds = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            if (j, i) not in bonds:  # Avoid duplicate bonds
                bonds.append((i, j))

    return bonds


def get_universe(file: Path) -> mda.Universe:
    """Create a MDAnalysis Universe from a H5MD file.

    Parameters
    ----------
    file : Path
        The path to the H5MD file.

    Returns
    -------
    mda.Universe
        The MDAnalysis Universe object.
    """

    if file.suffix not in [".h5", ".h5md", ".hdf5"]:
        raise ValueError("Currently, only HDF5 files are supported.")

    atoms: ase.Atoms = znh5md.IO(file)[0]
    # consider https://docs.mdanalysis.org/stable/documentation_pages/guesser_modules/default_guesser.html#MDAnalysis.guesser.default_guesser.DefaultGuesser.guess_bonds
    bonds = get_bonds(atoms)

    universe = mda.Universe.empty(n_atoms=len(atoms), trajectory=True)
    # https://userguide.mdanalysis.org/stable/examples/constructing_universe.html#Adding-topology-attributes
    universe.add_TopologyAttr("names", atoms.get_chemical_symbols())
    universe.add_TopologyAttr("type", atoms.get_chemical_symbols())
    universe.add_TopologyAttr("masses", atoms.get_masses())
    universe.add_TopologyAttr("bonds", bonds)

    # TODO: warning if the box changes in time! We do not support NPT yet!
    universe.dimensions = atoms.cell.cellpar()

    # add new unit translation
    # TODO: we assume these units, might not fit!
    H5MDReader._unit_translation["force"].update({"eV/Angstrom": "kJ/(mol*Angstrom)"})
    H5MDReader._unit_translation["velocity"].update({"Angstrom/fs": "Angstrom/fs"})

    reader = H5MDReader(file, convert_units=False)
    universe.trajectory = reader

    return universe
