from typing import Optional, List, Sequence, Callable
from dataclasses import dataclass
from functools import partial
from concurrent.futures import CancelledError

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs import ExplicitBitVect

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout

from orangewidget.utils.itemmodels import PyListModel
from orangewidget.utils.combobox import ComboBoxSearch, ComboBox

from Orange.data import Table, StringVariable, DiscreteVariable, Domain
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting

from orangecontrib.chem.widgets.common import OWConcurrentWidget, cbselect


@dataclass
class Result:
    data: Table


def fingerprint_from_smiles(
        smiles: str, fingerprint: Callable = Chem.RDKFingerprint
) -> List[float]:
    mol = Chem.MolFromSmiles(smiles)
    fp = fingerprint(mol)
    return list(fp)


def run(
        data,
        column: StringVariable,
        fingerprint: Callable,
        column_name_format: str,
        state: TaskState
) -> Result:
    smiles = data[:, column].metas.flatten()
    size = len(smiles)
    fps: List[Optional[Sequence[float]]] = [None] * size
    for i, sm in enumerate(smiles):
        if sm:
            try:
                res = fingerprint_from_smiles(sm, fingerprint)
            except Exception:
                pass
            else:
                fps[i] = res
        state.set_progress_value(100 * (i + 1) / size)
        if state.is_interruption_requested():
            raise CancelledError()
    data = table_concat_fingerprints(data, fps, column_name_format, 1)
    return Result(data)


def table_concat_fingerprints(
        table: Table,
        fps: Sequence[Optional[ExplicitBitVect]],
        name="F{}",
        start=1,
):
    if len(table) != len(fps):
        raise ValueError("len(table) != len(fps)")
    fp_count = max(map(len, filter(None, fps)), default=0)
    matrix: List[Optional[List[float]]] = [None] * len(fps)
    for i, fp in enumerate(fps):
        if fp is not None:
            matrix[i] = list(fp)
    NA = float("nan")
    for i in range(len(matrix)):
        if matrix[i] is None:
            matrix[i] = [NA] * fp_count
    fp_array = np.array(matrix)
    cols = [DiscreteVariable(name.format(i), values=("0", "1"))
            for i in range(start, fp_array.shape[1] + start)]
    extend = Table.from_numpy(Domain(cols), fp_array)
    return table.concatenate([table, extend], axis=1)


@dataclass
class FPMethod:
    key: str
    name: str
    func: Callable
    colname: str


Fingerprints = [
    FPMethod("RDKFingerprint", "RDK Fingerprint", Chem.RDKFingerprint, "RDF-{}"),
    FPMethod("GenMACCSKeys", "MACCS Keys", MACCSkeys.GenMACCSKeys, "MACCS-{}"),
    FPMethod("GetMorganFingerprint", "Morgan Fingerprint (ECFP4 like)",
             lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2),
             "M-{}"),
    FPMethod("GetMorganFingerprint_useFeatures",
             "Feature Morgan Fingerprint (FCFP4 like)",
             lambda mol: AllChem.GetMorganFingerprintAsBitVect(
                 mol, 2, useFeatures=True
             ),
             "M-{}")
]


class OWMoleculeFingerprints(OWConcurrentWidget):
    name = "Molecule Fingerprints"
    description = "Compute molecule fingerprints"
    icon = "../widgets/icons/category.svg"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False
    resizing_enabled = False

    smiles_var_name = Setting("SMILES")
    finger_print_method = Setting("RDKFingerprint")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: Optional[Table] = None
        self._smiles_var = None

        self.smiles_model = DomainModel(
            valid_types=(StringVariable,)
        )
        self.finger_print_model = model = PyListModel()
        for i, fp in enumerate(Fingerprints):
            model.append(fp.name)
            model.setItemData(
                model.index(i), {
                    Qt.UserRole: fp.func,
                    Qt.UserRole + 1: fp.colname,
                    Qt.UserRole + 2: fp.key,
                }
            )

        self.smiles_cb = ComboBoxSearch(
            minimumContentsLength=20,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon
        )
        self.smiles_cb.activated.connect(
            self.__set_smiles_index
        )
        self.finger_print_cb = ComboBox(
            minimumContentsLength=20,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon
        )
        self.finger_print_cb.activated.connect(
            self.__set_finger_print_index
        )
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )
        form.addRow("Smiles", self.smiles_cb)
        form.addRow("Fingerprints", self.finger_print_cb)
        self.controlArea.layout().addLayout(form)

        self.smiles_cb.setModel(self.smiles_model)
        self.finger_print_cb.setModel(self.finger_print_model)
        # Restore fingerprint
        cbselect(
            self.finger_print_cb, self.finger_print_method, Qt.UserRole + 2,
            default=0
        )

    def __set_smiles_index(self, index: int):
        self.smiles_cb.setCurrentIndex(index)
        if index < 0:
            smiles_var = None
        else:
            smiles_var = self.smiles_model[index]
        if self._smiles_var != smiles_var:
            self._smiles_var = smiles_var
            if smiles_var is not None:
                self.smiles_var_name = self._smiles_var.name
            self.invalidate()

    def __set_finger_print_index(self, index: int):
        self.finger_print_cb.setCurrentIndex(index)
        method = self.finger_print_cb.currentData(Qt.UserRole + 2)
        if self.finger_print_method != method:
            self.finger_print_method = method
            self.invalidate()

    @property
    def smiles_var(self):
        return self._smiles_var

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.invalidate()

    def handleNewSignals(self):
        if self.data is not None:
            self.smiles_model.set_domain(self.data.domain)
            idx = self.smiles_cb.findText(self.smiles_var_name)
            self.__set_smiles_index(idx)
        else:
            self.smiles_model.set_domain(None)
        super().handleNewSignals()

    def commit(self):
        if self.data is None or self.smiles_var is None:
            self.clear_outputs()
        else:
            self.start(self.create_task())

    def create_task(self):
        data = self.data
        column = self.smiles_var
        fp = self.finger_print_cb.currentData(Qt.UserRole)
        column_name_format = self.finger_print_cb.currentData(Qt.UserRole + 1)
        return partial(run, data, column, fp, column_name_format)

    def on_done(self, result: Result) -> None:
        self.Outputs.data.send(result.data)

    def clear_outputs(self):
        self.Outputs.data.send(None)