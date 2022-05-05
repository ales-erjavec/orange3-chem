import abc
import os
from typing import Any, Optional, TypedDict, Union

from AnyQt.QtCore import QTimer, Slot, Qt, QSettings
from AnyQt.QtWidgets import QComboBox, QFormLayout

from rdkit import RDLogger
from orangewidget import gui, settings
from orangewidget.utils.combobox import ComboBox, ComboBoxSearch

from Orange.data import StringVariable
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.utils import qname

Input = Input
Output = Output
Msg = Msg


class OWConcurrentWidget(OWWidget, ConcurrentWidgetMixin, openclass=True):
    auto_commit: bool = Setting(False)

    class Error(OWWidget.Error):
        unhandled_error = Msg("Unhandled exception")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__modified = False
        self.__commit_timer = QTimer(singleShot=True, interval=0)
        self.__commit_timer.timeout.connect(self.__do_commit)

        b = gui.auto_commit(
            self.buttonsArea, self, "auto_commit", "Apply",
            commit=lambda: self.__do_commit(),
            callback=lambda: self.__on_auto_commit_changed()
        )
        self.commit_button = b.button
        self.commit_button.setDefault(True)

    def __on_auto_commit_changed(self):
        if self.auto_commit and self.__modified:
            self.__do_commit()

    def __set_modified(self, state: bool):
        self.__modified = state
        font = self.commit_button.font()
        font.setItalic(state)
        self.commit_button.setFont(font)

    def invalidate(self):
        self.cancel()
        self.setInvalidated(True)
        self.__set_modified(True)
        if self.auto_commit:
            self.__commit_timer.start()

    @Slot()
    def __do_commit(self):
        self.cancel()
        self.__commit_timer.stop()
        self.__set_modified(False)
        self.Error.unhandled_error.clear()
        self.commit()

    @abc.abstractmethod
    def commit(self):
        return NotImplemented

    def on_exception(self, ex: Exception):
        self.Error.unhandled_error(exc_info=ex)
        super().on_exception(ex)

    def on_partial_result(self, result: Any) -> None:
        pass

    def handleNewSignals(self):
        super().handleNewSignals()
        self.__do_commit()


class SimpleFormWidget(OWConcurrentWidget, openclass=True):
    want_main_area = False
    resizing_enabled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RDLogger.DisableLog("rdApp.*")
        self.form = QFormLayout(
            objectName="main-form",
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
        )
        self.controlArea.layout().addLayout(self.form)


class SmilesFormWidget(SimpleFormWidget, openclass=True):
    smiles_var: Optional[StringVariable] = None

    class State(TypedDict):
        smiles_column: Optional[str]

    settings: State = Setting({
        "smiles_column": None,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = {
            **type(self).settings.default,
            **self.settings,
        }
        self.smiles_model = DomainModel()
        self.smiles_cb = ComboBoxSearch(
            objectName="smiles-cb",
            minimumContentsLength=20,
            sizeAdjustPolicy=ComboBoxSearch.AdjustToMinimumContentsLengthWithIcon
        )
        self.smiles_cb.setModel(self.smiles_model)
        self.smiles_cb.activated.connect(self.__set_smiles_index)
        self.form.addRow("Smiles", self.smiles_cb)

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

    def set_smiles_column(self, column: Union[StringVariable, int]):
        if isinstance(column, StringVariable):
            index = self.smiles_model.indexOf(column)
        else:
            index = column
        self.__set_smiles_index(index)

    def smiles_column(self) -> StringVariable:
        return self.smiles_var


class TextEditComboBox(ComboBox):
    def text(self) -> str:
        """
        Return the current text.
        """
        return self.itemText(self.currentIndex())

    def setText(self, text: str) -> None:
        """
        Set `text` as the current text (adding it to the model if necessary).
        """
        idx = self.findData(text, Qt.EditRole, Qt.MatchExactly)
        if idx != -1:
            self.setCurrentIndex(idx)
        else:
            self.addItem(text)
            self.setCurrentIndex(self.count() - 1)


def cbselect(
        cb: QComboBox, value, role: Qt.ItemDataRole = Qt.EditRole, default=-1
) -> None:
    """
    Find and select the `value` in the `cb` QComboBox.

    Parameters
    ----------
    cb: QComboBox
    value: Any
    role: Qt.ItemDataRole
        The data role in the combo box model to match value against.
    default: int
        The default index to set if value is not found.
    """
    idx = cb.findData(value, role)
    if idx == -1:
        idx = default
    cb.setCurrentIndex(idx)


def cb_find_smiles_column(cb: QComboBox, name=None, role=Qt.DisplayRole):
    if name is not None:
        return cb.findData(name, role)
    else:
        return cb.findData("smiles", role, Qt.MatchFixedString)


def local_settings(cls: type) -> QSettings:
    """Return a `QSettings` instance with local persistent settings."""
    filename = "{}.ini".format(qname(cls))
    fname = os.path.join(settings.widget_settings_dir(), filename)
    return QSettings(fname, QSettings.IniFormat)
