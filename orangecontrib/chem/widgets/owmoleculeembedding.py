import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QPushButton, QStyle

from Orange.data import Table, Variable
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
from Orange.widgets.gui import hBox
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.chem.molecule_embedder import MoleculeEmbedder
from orangecontrib.chem.molecule_embedder import MODELS as EMBEDDERS_INFO


class Result(SimpleNamespace):
    embedding: Optional[Table] = None
    skip_smiles: Optional[Table] = None
    num_skipped: int = None


def run_embedding(
    data: Table,
    smiles_column: Variable,
    embedder_name: str,
    state: TaskState,
) -> Result:
    """
    Run the embedding process

    Parameters
    ----------
    data
        Data table with smiles to embed.
    smiles_column
        The column of the table with smiles.
    embedder_name
        The name of selected embedder.
    state
        State object used for controlling and progress.

    Returns
    -------
    The object that holds embedded smiles, skipped smiles, and number
    of skipped smiles.
    """
    embedder = MoleculeEmbedder(model=embedder_name)

    file_paths = data[:, smiles_column].metas.flatten()

    file_paths_mask = file_paths == smiles_column.Unknown
    file_paths_valid = file_paths[~file_paths_mask]

    # init progress bar and fuction
    ticks = iter(np.linspace(0.0, 100.0, file_paths_valid.size))

    def advance(success=True):
        if state.is_interruption_requested():
            embedder.set_canceled()
        if success:
            state.set_progress_value(next(ticks))

    try:
        emb, skip, n_skip = embedder(
            data, col=smiles_column, callback=advance
        )
    except EmbeddingConnectionError:
        # recompute ticks to go from current state to 100
        ticks = iter(np.linspace(next(ticks), 100.0, file_paths_valid.size))

        state.set_partial_result("squeezenet")
        embedder = MoleculeEmbedder(model="smiles_cnn_local")
        emb, skip, n_skip = embedder(
            data, col=smiles_column, callback=advance
        )

    return Result(embedding=emb, skip_smiles=skip, num_skipped=n_skip)


class OWMoleculeEmbedding(OWWidget, ConcurrentWidgetMixin):
    name = "Molecule Embedding"
    description = "Molecule embedding through deep neural networks."
    keywords = ["embedding", "molecule embedding", "smiles"]

    icon = "../widgets/icons/category.svg"
    priority = 150

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Smiles", Table)

    class Outputs:
        embeddings = Output("Fingerprints", Table, default=True)
        skipped_smiles = Output("Skipped Smiles", Table)

    class Warning(OWWidget.Warning):
        switched_local_embedder = Msg(
            "No internet connection: switched to local embedder"
        )
        no_smiles_attribute = Msg(
            "Please provide data with an smiles attribute."
        )
        molecules_skipped = Msg("{} molecules are skipped.")

    class Error(OWWidget.Error):
        unexpected_error = Msg("Embedding error: {}")

    cb_smiles_attr_current_id: int = Setting(default=0)
    cb_embedder_current_id: int = Setting(default=0)

    _auto_apply: bool = Setting(default=True)

    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.embedders = sorted(list(EMBEDDERS_INFO),
                                key=lambda k: EMBEDDERS_INFO[k]['order'])
        self._string_attributes = None
        self._input_data = None
        self._log = logging.getLogger(__name__)
        self._task = None
        self._previous_attr_id = None
        self._previous_embedder_id = None

        self._setup_layout()

    def _setup_layout(self):
        widget_box = widgetBox(self.controlArea, 'Settings')
        self.cb_smiles_attr = comboBox(
            widget=widget_box,
            master=self,
            value='cb_smiles_attr_current_id',
            label='SMILES attribute:',
            orientation=Qt.Horizontal,
            callback=self._cb_smiles_attr_changed
        )

        self.cb_embedder = comboBox(
            widget=widget_box,
            master=self,
            value='cb_embedder_current_id',
            label='Embedder:',
            orientation=Qt.Horizontal,
            callback=self._cb_embedder_changed
        )
        names = [
            EMBEDDERS_INFO[e]["name"]
            + (" (local)" if EMBEDDERS_INFO[e].get("is_local") else "")
            for e in self.embedders
        ]
        self.cb_embedder.setModel(VariableListModel(names))
        if not self.cb_embedder_current_id < len(self.embedders):
            self.cb_embedder_current_id = 0
        self.cb_embedder.setCurrentIndex(self.cb_embedder_current_id)

        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info = widgetLabel(
            widget_box,
            EMBEDDERS_INFO[current_embedder]['description']
        )

        self.auto_commit_widget = auto_commit(
            widget=self.controlArea,
            master=self,
            value='_auto_apply',
            label='Apply',
            commit=self.commit
        )

        self.cancel_button = QPushButton(
            'Cancel',
            icon=self.style().standardIcon(QStyle.SP_DialogCancelButton),
        )
        self.cancel_button.clicked.connect(self.cancel)
        hbox = hBox(self.controlArea)
        hbox.layout().addWidget(self.cancel_button)
        self.cancel_button.setDisabled(True)

    def set_output_data_summary(self, data_emb, data_skip):
        if data_emb is None and data_skip is None:
            self.info.set_output_summary(self.info.NoOutput)
        else:
            success = 0 if data_emb is None else len(data_emb)
            skip = 0 if data_skip is None else len(data_skip)
            self.info.set_output_summary(
                f"{success}",
                f"{success} molecules successfully embedded ,\n"
                f"{skip} molecules skipped.",
            )

    @Inputs.data
    def set_data(self, data):
        self.Warning.clear()

        if not data:
            self._input_data = None
            self.clear_outputs()
            return

        self._string_attributes = MoleculeEmbedder.filter_string_attributes(data)
        if not self._string_attributes:
            input_data_info_text = (
                "Data with {:d} instances, but without string attributes."
                .format(len(data)))
            self.input_data_info.setText(input_data_info_text)
            self._input_data = None
            return

        if not self.cb_smiles_attr_current_id < len(self._string_attributes):
            self.cb_smiles_attr_current_id = 0

        self.cb_smiles_attr.setModel(VariableListModel(self._string_attributes))
        self.cb_smiles_attr.setCurrentIndex(self.cb_smiles_attr_current_id)

        self._input_data = data
        self._previous_attr_id = self.cb_smiles_attr_current_id
        self._previous_embedder_id = self.cb_embedder_current_id

        self.unconditional_commit()

    def _cb_smiles_attr_changed(self):
        self._cb_changed()

    def _cb_embedder_changed(self):
        self.Warning.switched_local_embedder.clear()
        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info.setText(
            EMBEDDERS_INFO[current_embedder]['description']
        )
        self._cb_changed()

    def _cb_changed(self):
        if (
            self._previous_embedder_id != self.cb_embedder_current_id
            or self._previous_attr_id != self.cb_smiles_attr_current_id
        ):
            # recompute embeddings only when selected value in dropdown changes
            self._previous_embedder_id = self.cb_embedder_current_id
            self._previous_attr_id = self.cb_smiles_attr_current_id
            self.cancel()
            self.commit()

    def commit(self):
        if not self._string_attributes or self._input_data is None:
            self.clear_outputs()
            return

        self.cancel_button.setDisabled(False)

        embedder_name = self.embedders[self.cb_embedder_current_id]
        smiles_attribute = self._string_attributes[self.cb_smiles_attr_current_id]
        self.start(
            run_embedding, self._input_data, smiles_attribute, embedder_name
        )
        self.Error.unexpected_error.clear()

    def on_done(self, result: Result) -> None:
        """
        Invoked when task is done.

        Parameters
        ----------
        result
            Embedding results.
        """
        self.cancel_button.setDisabled(True)
        assert len(self._input_data) == len(result.embedding or []) + len(
            result.skip_smiles or []
        )
        self._send_output_signals(result)

    def on_partial_result(self, result: str) -> None:
        self._switch_to_local_embedder()

    def on_exception(self, ex: Exception) -> None:
        """
        When an exception occurs during the calculation.

        Parameters
        ----------
        ex
            Exception occurred during the embedding.
        """
        log = logging.getLogger(__name__)
        log.debug(ex, exc_info=ex)
        self.cancel_button.setDisabled(True)
        self.Error.unexpected_error(type(ex).__name__, exc_info=ex)
        self.clear_outputs()
        logging.debug("Exception", exc_info=ex)

    def _switch_to_local_embedder(self):
        self.Warning.switched_local_embedder()
        self.cb_embedder_current_id = self.embedders.index("smiles_cnn_local")

    def _send_output_signals(self, result: Result) -> None:
        self.Warning.molecules_skipped.clear()
        self.Outputs.embeddings.send(result.embedding)
        self.Outputs.skipped_smiles.send(result.skip_smiles)
        if result.num_skipped != 0:
            self.Warning.molecules_skipped(result.num_skipped)
        self.set_output_data_summary(result.embedding, result.skip_smiles)

    def clear_outputs(self):
        self._send_output_signals(
            Result(embedding=None, skip_smiles=None, num_skipped=0)
        )

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()
