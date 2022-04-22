from functools import partial
from typing import Optional, Callable, Sequence

from AnyQt.QtWidgets import QCheckBox

from rdkit import Chem
from rdkit.Chem import rdFMCS

from orangewidget.settings import Setting
from orangewidget.utils.signals import Output

from Orange.data import Table, StringVariable
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.spinbox import DoubleSpinBox

from orangecontrib.chem.widgets.common import (
    SmilesFormWidget, Input, cb_find_smiles_column
)


class OWMCS(SmilesFormWidget):
    name = "Maximum Common Substructure"
    description = "Find Maximum Common Substructure"
    keywords = ["maximum", "common", "substructure"]
    icon = "icons/category.svg"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        SMARTS = Output("SMARTS", str)

    class State(SmilesFormWidget.State):
        threshold: float
        complete_rings_only: bool
        ring_matches_ring_only: bool
        match_valences: bool
        match_chiral_tag: bool

    settings: State = Setting({
        **SmilesFormWidget.settings.default,
        "threshold": 1.0,
        "complete_rings_only": False,
        "ring_matches_ring_only": False,
        "match_valences": False,
        "match_chiral_tag": False
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: Optional[Table] = None

        self.threshold_spin = DoubleSpinBox(
            minimum=0, maximum=1.0, singleStep=0.05, decimals=3,
            value=self.settings["threshold"]
        )
        self.threshold_spin.valueChanged.connect(self.set_threshold)
        self.complete_rings_only_cb = QCheckBox(
            "Ring matches ring only",
            checked=self.settings["complete_rings_only"],
        )
        self.ring_matches_ring_only_cb = QCheckBox(
            "Match complete rings",
            checked=self.settings["ring_matches_ring_only"],
        )
        self.match_valences_cb = QCheckBox(
            "Match valences",
            checked=self.settings["match_valences"],
        )
        self.match_chiral_tag_cb = QCheckBox(
            "Match Chirality",
            checked=self.settings["match_chiral_tag"]
        )
        self.complete_rings_only_cb.toggled.connect(
            self.set_complete_rings_only
        )
        self.ring_matches_ring_only_cb.toggled.connect(
            self.set_ring_matches_ring_only
        )
        self.match_valences_cb.toggled.connect(
            self.set_match_valences
        )
        self.match_chiral_tag_cb.toggled.connect(
            self.set_match_chiral_tag
        )
        self.form.addRow("Threshold", self.threshold_spin)
        self.form.addRow("", self.complete_rings_only_cb)
        self.form.addRow("", self.ring_matches_ring_only_cb)
        self.form.addRow("", self.match_valences_cb)
        self.form.addRow("", self.match_chiral_tag_cb)

    def set_threshold(self, threshold: float):
        if threshold != self.settings["threshold"]:
            self.threshold_spin.setValue(threshold)
            self.settings["threshold"] = threshold
            self.invalidate()

    def threshold(self) -> float:
        return self.settings["threshold"]

    def set_complete_rings_only(self, match: bool):
        if match != self.settings["complete_rings_only"]:
            self.complete_rings_only_cb.setChecked(match)
            self.settings["complete_rings_only"] = match
            self.invalidate()

    def complete_rings_only(self):
        return self.settings["complete_rings_only"]

    def set_ring_matches_ring_only(self, state: bool):
        if state != self.settings["ring_matches_ring_only"]:
            self.ring_matches_ring_only_cb.setChecked(state)
            self.settings["ring_matches_ring_only"] = state
            self.invalidate()

    def ring_matches_ring_only(self):
        return self.settings["ring_matches_ring_only"]

    def set_match_valences(self, state: bool):
        if state != self.settings["match_valences"]:
            self.match_valences_cb.setChecked(state)
            self.settings["match_valences"] = state
            self.invalidate()

    def match_valences(self):
        return self.settings["match_valences"]

    def set_match_chiral_tag(self, state: bool):
        if state != self.settings["match_chiral_tag"]:
            self.match_chiral_tag_cb.setChecked(state)
            self.settings["match_chiral_tag"] = state
            self.invalidate()

    def match_chiral_tag(self):
        return self.settings["match_chiral_tag"]

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.smiles_model.set_domain(data.domain)
            idx = cb_find_smiles_column(
                self.smiles_cb, self.settings["smiles_column"]
            )
            self.set_smiles_column(idx)
        else:
            self.smiles_model.set_domain(None)
        self.invalidate()

    def handleNewSignals(self):
        super().handleNewSignals()

    def commit(self):
        if self.data is not None and self.smiles_var is not None:
            self.start(self.create_task())
        else:
            self.clear_outputs()

    def create_task(self):
        data = self.data
        smiles_var = self.smiles_var

        return partial(
            run, data, smiles_var,
            threshold=self.settings["threshold"],
            complete_rings_only=self.settings["complete_rings_only"],
            ring_matches_ring_only=self.settings["ring_matches_ring_only"],
            match_valences=self.settings["match_valences"],
            match_chiral_tag=self.settings["match_chiral_tag"],
        )

    def on_done(self, result) -> None:
        self.Outputs.SMARTS.send(result.smartsString)

    def clear_outputs(self):
        self.Outputs.SMARTS.send(None)


def run(
        data: Table,
        smiles_column: StringVariable,
        state: TaskState,
        threshold=1.0,
        complete_rings_only=False,
        ring_matches_ring_only=False,
        match_valences=False,
        match_chiral_tag=False,
):
    smiles = data[:, smiles_column].metas.flatten().tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = list(filter(None, mols))

    def pb(f):
        state.set_progress_value(f)
        return not state.is_interruption_requested()

    res = find_mcs(
        mols,
        threshold=threshold,
        complete_rings_only=complete_rings_only,
        ring_matches_ring_only=ring_matches_ring_only,
        match_valences=match_valences,
        match_chiral_tag=match_chiral_tag,
        callback=pb,
    )
    return res


class _ProgressCallback(rdFMCS.MCSProgress):

    def __init__(self, callback: Callable[[float], bool]):
        super().__init__()
        self._callback = callback

    def __call__(self, stat, params):
        n = stat.seedProcessed
        return self._callback(n / 10000)


def find_mcs(
        mols: Sequence[Chem.Mol],
        threshold=1.0,
        complete_rings_only=False,
        ring_matches_ring_only=False,
        match_valences=False,
        match_chiral_tag=False,
        atom_comp=rdFMCS.AtomCompare.CompareElements,
        bond_comp=rdFMCS.BondCompare.CompareOrder,
        ring_comp=rdFMCS.RingCompare.IgnoreRingFusion,
        timeout=3600,
        callback=None,

):
    ps = rdFMCS.MCSParameters()
    ps.MaximizeBonds = True
    ps.Threshold = threshold
    ps.Timeout = timeout

    ps.AtomTyper = atom_comp
    ps.AtomCompareParameters.MatchValences = match_valences
    ps.AtomCompareParameters.MatchChiralTag = match_chiral_tag
    ps.AtomCompareParameters.RingMatchesRingOnly = ring_matches_ring_only

    ps.BondTyper = bond_comp
    ps.BondCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    ps.BondCompareParameters.CompleteRingsOnly = complete_rings_only
    ps.BondCompareParameters.MatchFusedRings = \
        (ring_comp != rdFMCS.RingCompare.IgnoreRingFusion)
    ps.BondCompareParameters.MatchFusedRingsStrict = \
        (ring_comp == rdFMCS.RingCompare.StrictRingFusion)

    if callback is not None:
        ps.ProgressCallback = _ProgressCallback(callback)

    res = rdFMCS.FindMCS(mols, ps)
    return res

