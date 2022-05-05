from Orange.widgets.tests.base import WidgetTest
from orangecontrib.chem.widgets.owmoleculefingerprints import \
    OWMoleculeFingerprints
from orangecontrib.chem.widgets.tests import test_data


class TestOWMoleculeFingerprints(WidgetTest):
    def setUp(self) -> None:
        super().setUp()
        self.widget = self.create_widget(OWMoleculeFingerprints)
        self.data = test_data()

    def test_widget(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        out = self.get_output(w.Outputs.data)
        self.assertGreaterEqual(len(out.domain.attributes), 10)
