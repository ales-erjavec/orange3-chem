from functools import partial
from typing import Optional, TypedDict
from concurrent.futures import CancelledError

import numpy as np
from rdkit import Chem

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout

from orangecanvas.utils.settings import (
    QSettings_readArray, QSettings_writeArray
)

from orangewidget.settings import Setting
from orangewidget.utils.combobox import ComboBoxSearch
from orangewidget.utils.itemmodels import PyListModel

from Orange.data import Table, StringVariable
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)

from orangecontrib.chem.widgets.common import (
    OWConcurrentWidget, Input, Output, Msg, TextEditComboBox, local_settings,
)

MAX_HISTORY = 30


class OWMoleculeFilter(OWConcurrentWidget):
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

    want_main_area = False
    resizing_enabled = False

    class State(TypedDict):
        smiles_column: str
        filter_pattern: str

    settings: State = Setting({
        "smiles_column": "SMILES",
        "filter_pattern": "C(=O)[O;H,-]"
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: Optional[Table] = None
        self.smiles_var = None
        self.smiles_model = DomainModel(valid_types=(StringVariable,))
        self.filter_model = PyListModel()
        self.smiles_cb = ComboBoxSearch(
            minimumContentsLength=20,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon,
        )
        self.filter_cb = TextEditComboBox(
            minimumContentsLength=40,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon,
            insertPolicy=TextEditComboBox.InsertAtTop,
            editable=True,
        )
        self.filter_cb.lineEdit().setPlaceholderText("SMARTS pattern...")
        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )
        form.addRow("SMILES", self.smiles_cb)
        form.addRow("Filter", self.filter_cb)

        self.controlArea.layout().addLayout(form)
        self.smiles_cb.setModel(self.smiles_model)
        self.filter_cb.setModel(self.filter_model)
        self.smiles_cb.activated.connect(self.__set_smiles_index)
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

    def __set_smiles_index(self, index: int):
        self.smiles_cb.setCurrentIndex(index)
        if index < 0:
            smiles_var = None
        else:
            smiles_var = self.smiles_model[index]

        if self.smiles_var != smiles_var:
            self.smiles_var = smiles_var
            if smiles_var is not None:
                self.settings["smiles_column"] = smiles_var.name
            self.invalidate()

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

    @property
    def current_smiles(self):
        idx = self.smiles_cb.currentIndex()
        if idx >= 0:
            return self.smiles_model[idx]
        return None

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
            idx = self.smiles_cb.findText(self.settings["smiles_column"])
            self.__set_smiles_index(idx)
        else:
            self.smiles_model.set_domain(None)

        super().handleNewSignals()

    def commit(self):
        self.Error.smarts.clear()
        if self.data is None or self.current_smiles is None:
            self.clear_outputs()
        else:
            self.start(self.create_task())

    def create_task(self):
        data = self.data
        column = self.current_smiles
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


def run(data, column, pattern, state: TaskState):
    coldata = data[:, column].metas.flatten()
    patt = Chem.MolFromSmarts(pattern)
    if patt is None:
        raise InvalidSmartsPattern()
    N = len(coldata)
    matches = [False] * N
    for i, smiles in enumerate(coldata):
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            matches[i] = False
        else:
            match = mol.GetSubstructMatch(patt)
            matches[i] = bool(match)
        state.set_progress_value(100. * i // N)
        if state.is_interruption_requested():
            raise CancelledError()

    matches = np.array(matches, bool)
    selected = data[matches]
    annotated = create_annotated_table(data, matches)
    return selected, annotated
