from AnyQt.QtTest import  QTest

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.chem.widgets.owmoleculeviewer import OWMoleculeViewer
from orangecontrib.chem.widgets.tests import test_data


class TestOWMoleculeViewer(WidgetTest):
    def setUp(self) -> None:
        super().setUp()
        self.widget = self.create_widget(
            OWMoleculeViewer
        )
        self.data = test_data()

    def test_widget(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        view = self.widget.thumbnailView
        model = view.model()
        self.assertEqual(view.model().rowCount(), len(self.data))
        view.grab()
        QTest.qWait(100)
        smodel = view.selectionModel()
        smodel.select(model.index(1, 0), smodel.ClearAndSelect)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[:, "SMILES"].metas.flatten()[0], "Br")

        self.send_signal(self.widget.Inputs.data, None)
        out = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(out)


