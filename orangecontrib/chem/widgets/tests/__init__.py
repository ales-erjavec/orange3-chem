import numpy as np

from Orange.data import Table, Domain, StringVariable


def test_data():
    return Table.from_numpy(
        Domain([], [], [StringVariable("SMILES")]),
        np.empty((3, 0)), None,
        [["CC"],
         ["Br"],
         ["c1ccccc1"]],
    )
