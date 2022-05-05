from Orange.widgets.tests.base import WidgetTest
from orangecontrib.chem.widgets.owmcs import OWMCS
from orangecontrib.chem.widgets.tests import test_data


class TestOWMCS(WidgetTest):
    def setUp(self) -> None:
        super().setUp()
        self.widget = self.create_widget(OWMCS)
        self.data = test_data()

    def test_widget(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        out = self.get_output(w.Outputs.SMARTS)
        self.assertEqual(out, "")
        self.send_signal(w.Inputs.data, None)
        out = self.get_output(w.Outputs.SMARTS)
        self.assertEqual(out, None)
