from functools import partial
from typing import Optional
from concurrent.futures import CancelledError, Executor

import numpy as np
from rdkit import Chem

from AnyQt.QtCore import Qt

from orangecanvas.utils.settings import (
    QSettings_readArray, QSettings_writeArray
)

from orangewidget.settings import Setting
from orangewidget.utils.combobox import ComboBoxSearch
from orangewidget.utils.itemmodels import PyListModel

from Orange.data import Table
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)
from orangecontrib.chem.widgets.common import (
    OWConcurrentWidget, Input, Output, Msg, TextEditComboBox, local_settings,
    SmilesFormWidget, ProcessPoolWidget, cb_find_smiles_column,
)

MAX_HISTORY = 30


class OWMoleculeFilter(SmilesFormWidget, ProcessPoolWidget):
    name = "Molecule Filter"
    description = "Filter molecules based on SMARTS patterns"
    icon = "icons/category.svg"

    class Inputs:
        data = Input("Data", Table)
        smarts = Input("SMARTS", str)

    class Outputs:
        selected_data = Output("Selected Data", Table)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Error(OWConcurrentWidget.Error):
        smarts = Msg("Invalid SMARTS pattern")

    class State(SmilesFormWidget.State):
        filter_pattern: str

    settings: State = Setting({
        **SmilesFormWidget.settings.default,
        "filter_pattern": "C(=O)[O;H,-]"
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: Optional[Table] = None
        self.filter_model = PyListModel()
        self.filter_cb = TextEditComboBox(
            minimumContentsLength=40,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon,
            insertPolicy=TextEditComboBox.InsertAtTop,
            editable=True,
        )
        self.filter_cb.lineEdit().setPlaceholderText("SMARTS pattern...")
        self.form.addRow("Filter", self.filter_cb)

        self.filter_cb.setModel(self.filter_model)
        self.filter_cb.activated.connect(self.__filter_changed)
        self.restore()
        # TODO: Pre-populate filter_model with
        #  rdkit/Data/Functional_Group_Hierarchy.txt

    @classmethod
    def local_settings(cls):
        return local_settings(cls)

    RecentSmartsSchema = {
        "key": str,
        "pattern": str,
    }

    def restore(self):
        model = self.filter_model
        settings = self.local_settings()
        saved = self.settings["filter_pattern"]
        recent_items = QSettings_readArray(
            settings, "recent", {
                "key": str,
                "pattern": str,
            }
        )
        recent_items = [item for item in recent_items if item["pattern"] != saved]
        if saved:
            recent_items.insert(0, {"key": "", "pattern": saved})
        for item in recent_items:
            model.append(item["pattern"])
            model.setData(model.index(model.rowCount() -1), item["key"], Qt.UserRole)
        self.filter_cb.setCurrentIndex(0)

    def __filter_changed(self):
        smarts = self.filter_cb.currentText()
        if smarts and Chem.MolFromSmarts(smarts):
            self.settings["filter_pattern"] = smarts
            self._note_recent_smarts(smarts)
            self.invalidate()

    def _note_recent_smarts(self, smarts: str):
        # store item to local persistent settings
        s = self.local_settings()
        arr = QSettings_readArray(s, "recent", self.RecentSmartsSchema)
        item = {"pattern": smarts, "key": ""}
        arr = [item for item in arr if item["pattern"] != smarts]
        arr.insert(0, item)
        arr = arr[:MAX_HISTORY]
        QSettings_writeArray(s, "recent", arr)

    @property
    def current_filter(self) -> str:
        return self.filter_cb.currentData(Qt.EditRole)

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.invalidate()

    @Inputs.smarts
    def set_smarts(self, smarts):
        if smarts != self.current_filter:
            if smarts:
                self.filter_cb.setText(smarts)
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
        self.Error.smarts.clear()
        if self.data is None or self.smiles_column() is None:
            self.clear_outputs()
        else:
            self.start(self.create_task())

    def create_task(self):
        data = self.data
        column = self.smiles_column()
        pattern = self.current_filter
        return partial(run, data, column, pattern)

    def on_done(self, result) -> None:
        selected, annotated = result
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def on_exception(self, ex: Exception):
        if isinstance(ex, InvalidSmartsPattern):
            self.Error.smarts()
        else:
            self.Error.unhandled_error(exc_info=ex)

    def clear_outputs(self):
        self.Outputs.selected_data.send(None)


class InvalidSmartsPattern(ValueError):
    pass


def matches(smi: str, patt: Chem.Mol) -> bool:
    if not smi:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    match = mol.GetSubstructMatch(patt)
    return bool(match)


def run(data, column, pattern,  executor: Executor, state: TaskState):
    coldata = data[:, column].metas.flatten().tolist()
    N = len(coldata)
    patt = Chem.MolFromSmarts(pattern)
    if patt is None:
        raise InvalidSmartsPattern()
    res_arr = [False] * N
    res = executor.map(
        partial(matches, patt=patt), coldata, chunksize=512
    )
    try:
        for i, match in enumerate(res):
            res_arr[i] = bool(match)
            state.set_progress_value(100. * i // N)
            if state.is_interruption_requested():
                raise CancelledError()
    finally:
        res.close()

    res_arr = np.array(res_arr, bool)
    selected = data[res_arr]
    annotated = create_annotated_table(data, res_arr)
    return selected, annotated
