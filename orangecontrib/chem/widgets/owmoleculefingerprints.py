from types import GeneratorType
from typing import Optional, List, Sequence, Callable
from dataclasses import dataclass
from functools import partial
from concurrent.futures import CancelledError, Executor

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs import ExplicitBitVect

import numpy as np

from AnyQt.QtCore import Qt

from orangewidget.utils.itemmodels import PyListModel
from orangewidget.utils.combobox import ComboBoxSearch, ComboBox

from Orange.data import Table, StringVariable, DiscreteVariable, Domain
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting

from orangecontrib.chem.widgets.common import (
    SmilesFormWidget, cbselect, cb_find_smiles_column, ProcessPoolWidget
)


@dataclass
class Result:
    data: Table


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


def fingerprints(smiles: Sequence[str], fingerprint, progress, executor: Executor):
    size = len(smiles)
    fps: List[Optional[Sequence[float]]] = [None] * size
    Fs = executor.map(
        partial(fingerprint_from_smiles, fingerprint=fingerprint),
        smiles,
        chunksize=127,
    )
    assert isinstance(Fs, GeneratorType)
    try:
        for i, res in enumerate(Fs):
            fps[i] = res
            progress(100 * (i + 1) / size)
    finally:
        Fs.close()
    return fps


def run(
        data,
        column: StringVariable,
        fingerprint: Callable,
        column_name_format: str,
        executor: Executor,
        state: TaskState
) -> Result:
    def progress(value):
        state.set_progress_value(value)
        if state.is_interruption_requested():
            raise CancelledError
    smiles = data[:, column].metas.flatten().tolist()
    fps = fingerprints(smiles, fingerprint, progress, executor)
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


def RDKFingerprint(mol):
    return Chem.RDKFingerprint(mol)


def GenMACCSKeys(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def GetMorganFingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2),


def GetMorganFingerprint_useFeatures(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)


@dataclass
class FPMethod:
    key: str
    name: str
    func: Callable
    colname: str


Fingerprints = [
    FPMethod("RDKFingerprint", "RDK Fingerprint", RDKFingerprint, "RDF-{}"),
    FPMethod("GenMACCSKeys", "MACCS Keys", GenMACCSKeys, "MACCS-{}"),
    FPMethod("GetMorganFingerprint", "Morgan Fingerprint (ECFP4 like)",
             GetMorganFingerprint,
             "M-{}"),
    FPMethod("GetMorganFingerprint_useFeatures",
             "Feature Morgan Fingerprint (FCFP4 like)",
             GetMorganFingerprint_useFeatures,
             "M-{}")
]


class OWMoleculeFingerprints(SmilesFormWidget, ProcessPoolWidget):
    name = "Molecule Fingerprints"
    description = "Compute molecule fingerprints"
    icon = "../widgets/icons/category.svg"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class State(SmilesFormWidget.State):
        finger_print_method: str

    settings: State = Setting({
        **SmilesFormWidget.settings.default,
        "finger_print_method": "RDKFingerprint",
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: Optional[Table] = None
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
        self.finger_print_cb = ComboBox(
            minimumContentsLength=20,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon
        )
        self.finger_print_cb.activated.connect(
            self.__set_finger_print_index
        )

        self.form.addRow("Fingerprints", self.finger_print_cb)
        self.finger_print_cb.setModel(self.finger_print_model)
        # Restore fingerprint
        cbselect(
            self.finger_print_cb, self.settings["finger_print_method"],
            Qt.UserRole + 2, default=0
        )

    def __set_finger_print_index(self, index: int):
        self.finger_print_cb.setCurrentIndex(index)
        method = self.finger_print_cb.currentData(Qt.UserRole + 2)
        if self.settings["finger_print_method"] != method:
            self.settings["finger_print_method"] = method
            self.invalidate()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.invalidate()

    def handleNewSignals(self):
        if self.data is not None:
            self.smiles_model.set_domain(self.data.domain)
            idx = cb_find_smiles_column(
                self.smiles_cb, self.settings["smiles_column"]
            )
            self.set_smiles_column(idx)
        else:
            self.smiles_model.set_domain(None)
        super().handleNewSignals()

    def commit(self):
        if self.data is None or self.smiles_column() is None:
            self.clear_outputs()
        else:
            self.start(self.create_task())

    def create_task(self):
        data = self.data
        column = self.smiles_column()
        fp = self.finger_print_cb.currentData(Qt.UserRole)
        column_name_format = self.finger_print_cb.currentData(Qt.UserRole + 1)
        return partial(run, data, column, fp, column_name_format)

    def on_done(self, result: Result) -> None:
        self.Outputs.data.send(result.data)

    def clear_outputs(self):
        self.Outputs.data.send(None)
