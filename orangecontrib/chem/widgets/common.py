import abc
from typing import Any

from AnyQt.QtCore import QTimer, Slot, Qt
from AnyQt.QtWidgets import QComboBox

from orangewidget import gui

from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

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