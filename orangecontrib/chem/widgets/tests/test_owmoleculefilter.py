from Orange.widgets.tests.base import WidgetTest
from orangecontrib.chem.widgets.owmoleculefilter import OWMoleculeFilter
from orangecontrib.chem.widgets.tests import test_data


class TestOWMoleculeFilter(WidgetTest):
    def setUp(self) -> None:
        super().setUp()
        self.widget = self.create_widget(OWMoleculeFilter)
        self.data = test_data()

    def test_widget(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        out = self.get_output(w.Outputs.selected_data)
        self.assertEqual(len(out), 0)
        self.send_signal(w.Inputs.smarts, "C")
        out = self.get_output(w.Outputs.selected_data)
        self.assertEqual(len(out), 1)
