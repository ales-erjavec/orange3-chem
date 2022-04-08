"""
Molecule Image Viewer Widget
-------------------

"""
from collections import namedtuple
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Sequence, Optional

from rdkit import Chem
from rdkit.Chem import Draw

from AnyQt.QtGui import QKeyEvent, QPainter, QIcon
from AnyQt.QtWidgets import QApplication, QStyleOptionViewItem, QStyle
from AnyQt.QtSvg import QSvgRenderer
from AnyQt.QtCore import (
    Qt, QSize, QModelIndex, QObject, QEvent, Signal, Slot,
    QPersistentModelIndex
)
from orangecanvas.gui.svgiconengine import SvgIconEngine
from orangewidget.utils.cache import LRUCache
from orangewidget.utils.concurrent import FutureWatcher
from orangewidget.utils.itemmodels import PyListModel

import Orange.data
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Output

from orangecontrib.imageanalytics.widgets.utils import (
    thumbnailview, imagepreview
)


_ImageItem = namedtuple(
    "_ImageItem",
    ["index",   # Row index in the input data table
     "smiles",  # Molecule SMILES
])


def svg_from_smiles(s: str, size=(400, 400)) -> bytes:
    p = Chem.MolFromSmiles(s)
    if p:
        draw = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
        draw.DrawMolecule(p)
        draw.FinishDrawing()
        svg = draw.GetDrawingText()
        return svg.encode("utf-8")
    else:
        return b''


SmilesDataRole = Qt.UserRole + 100
SvgDataRole = SmilesDataRole + 1


class MoleculesView(thumbnailview.IconView):
    currentIndexChanged = Signal(QModelIndex)

    def __init__(self, *args, **kwargs):
        super(MoleculesView, self).__init__(*args, **kwargs)
        preview = MolPreview()
        preview.setParent(self, Qt.WindowStaysOnTopHint | Qt.Tool)
        preview.setAttribute(Qt.WA_ShowWithoutActivating)
        preview.setVisible(False)
        preview.setIconView(self)
        self.preview = preview

    def currentChanged(self, current: QModelIndex, previous: QModelIndex) -> None:
        self.currentIndexChanged.emit(current)
        super().currentChanged(current, previous)


class MoleculeViewDelegate(thumbnailview.IconViewDelegate):
    def __init__(self, *args, **kwargs):
        super(MoleculeViewDelegate, self).__init__(*args, **kwargs)
        self._svg_renderer_cache = LRUCache(maxlen=1000)
        self._exc = ThreadPoolExecutor()

    def renderThumbnail(self, index: QModelIndex) -> 'Future':
        smiles = index.data(SmilesDataRole)
        pindex = QPersistentModelIndex(index)

        def f():
            svg = None
            if isinstance(smiles, str):
                svg = svg_from_smiles(smiles)
            return pindex, svg
        f = self._exc.submit(f)
        w = FutureWatcher(f, done=self.__on_svg_finished)
        f.watcher = w
        return f

    @Slot(Future)
    def __on_svg_finished(self, f):
        pindex, content = f.result()
        if pindex.isValid():
            index = QModelIndex(pindex)
            model = index.model()
            model.setData(index, content, SvgDataRole)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        sh = super().sizeHint(option, index)
        icsize = option.decorationSize
        maxw = icsize.width()
        sh.setWidth(min(sh.width(), int(maxw + 8)))
        return sh

    def paint(self, painter, option, index) -> None:
        self.startThumbnailRender(index)
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        svg = index.data(SvgDataRole)

        if svg is not None:
            eng = self._svg_renderer_cache.get(svg)
            if eng is None:
                eng = SvgIconEngine(svg)
            opt.icon = QIcon(eng)
            opt.features |= QStyleOptionViewItem.HasDecoration
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)


class MolPreview(imagepreview.Preview):
    widget: MoleculesView = None
    _svg = None
    _renderer = None

    def setIconView(self, view: thumbnailview.IconView):
        if self.widget is not None:
            self.widget.currentIndexChanged.disconnect(self.updatePreview)
            self.widget.removeEventFilter(self)
        self.widget = view
        if self.widget is not None:
            self.widget.currentIndexChanged.connect(self.updatePreview)
            self.widget.installEventFilter(self)

    def iconView(self):
        return self.widget

    def eventFilter(self, object: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape and self.isVisible():
            self.close()
            event.accept()
            return True
        elif event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            self.toggleVisible()
            event.accept()
            return True
        elif event.type() == QEvent.Hide:
            self.close()
        return super().eventFilter(object, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Space, Qt.Key_Escape):
            self.close()
            event.accept()
            return
        elif event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            QApplication.sendEvent(self.widget.viewport(), event)
            return
        super().keyPressEvent(event)

    def toggleVisible(self):
        index = self.widget.currentIndex()
        if index.isValid():
            self.setVisible(True)
            self.raise_()
            self.updatePreview()
        else:
            self.close()

    def updatePreview(self):
        index = self.widget.currentIndex()
        delegate = self.widget.itemDelegate()
        assert isinstance(delegate, MoleculeViewDelegate)
        svg = index.data(SvgDataRole)
        if svg is not None:
            self.setSvgContent(svg)
        else:
            self.setSvgContent(b'')

    def setSvgContent(self, svg):
        self._svg = svg
        self._renderer = QSvgRenderer(svg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self._renderer.render(painter)

    def sizeHint(self):
        if self._renderer:
            return self._renderer.defaultSize()
        else:
            return super().sizeHint()


class OWMoleculeViewer(widget.OWWidget):
    name = "Molecule Viewer"
    description = "View molecules referred to in the data."
    icon = "../widgets/icons/MoleculeViewer.svg"
    priority = 105

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    settingsHandler = settings.DomainContextHandler()

    smilesAttr = settings.ContextSetting(0)
    titleAttr = settings.ContextSetting(0)

    imageSize = settings.Setting(100)
    autoCommit = settings.Setting(True)

    buttons_area_orientation = Qt.Vertical
    graph_name = "thumbnailView"

    UserAdviceMessages = [
        widget.Message(
            "Pressing the 'Space' key while the thumbnail view has focus and "
            "a selected item will open a window with a full image",
            persistent_id="preview-introduction")
    ]

    def __init__(self):
        super().__init__()
        self.data = None
        self.allAttrs = []
        self.stringAttrs = []

        self.selectedIndices = []

        self.items: List[_ImageItem] = []

        self.smilesAttrCB = gui.comboBox(
            self.controlArea, self, "smilesAttr",
            box="SMILES Attribute",
            tooltip="Attribute with SMILES values",
            callback=[self.clearModel, self.setupModel],
            contentsLength=12,
            addSpace=True
        )

        self.titleAttrCB = gui.comboBox(
            self.controlArea, self, "titleAttr",
            box="Title Attribute",
            tooltip="Attribute with image title",
            callback=self.updateTitles,
            contentsLength=12,
            addSpace=True
        )

        gui.hSlider(
            self.controlArea, self, "imageSize",
            box="Image Size", minValue=32, maxValue=1024, step=16,
            callback=self.updateSize,
            createLabel=False
        )
        gui.rubber(self.controlArea)

        gui.auto_commit(self.buttonsArea, self, "autoCommit", "Send", box=False)

        self.thumbnailView = MoleculesView(
            resizeMode=MoleculesView.Adjust,
            iconSize=QSize(self.imageSize, self.imageSize),
            layoutMode=MoleculesView.Batched,
            wordWrap=True,
            batchSize=200,
        )
        self.thumbnailView.setTextElideMode(Qt.ElideMiddle)
        self.delegate = MoleculeViewDelegate()
        self.thumbnailView.setItemDelegate(self.delegate)
        self.mainArea.layout().addWidget(self.thumbnailView)

    def sizeHint(self):
        return QSize(800, 600)

    @Inputs.data
    def setData(self, data):
        self.closeContext()
        self.clear()

        self.data = data

        if data is not None:
            domain = data.domain
            self.allAttrs = (domain.class_vars + domain.metas +
                             domain.attributes)

            self.stringAttrs = [a for a in domain.metas if a.is_string]

            self.stringAttrs = sorted(
                self.stringAttrs,
                key=lambda attr: 0 if str(attr.name).lower() == "smiles" else 1
            )

            # If there are columns annotated with SMILES or CID,they are taken as default values
            # for smilesattr and titleAttr
            smiles_cols = [i for i, var in enumerate(self.stringAttrs)
                           if str(var.name).lower() == "smiles"]
            if smiles_cols:
                self.smilesAttr = smiles_cols[0]

            cid_cols = [i for i, var in enumerate(self.allAttrs)
                        if str(var.name).lower() == "cid"]
            if cid_cols:
                self.titleAttr = cid_cols[0]

            self.smilesAttrCB.setModel(VariableListModel(self.stringAttrs))
            self.titleAttrCB.setModel(VariableListModel(self.allAttrs))

            self.openContext(data)

            self.smilesAttr = max(min(self.smilesAttr, len(self.stringAttrs) - 1), 0)
            self.titleAttr = max(min(self.titleAttr, len(self.allAttrs) - 1), 0)

            if self.stringAttrs:
                self.setupModel()

    def clear(self):
        self.data = None
        self.error()
        self.smilesAttrCB.clear()
        self.titleAttrCB.clear()
        self.clearModel()

    def setupModel(self):
        self.error()
        if self.data is not None:
            attr = self.stringAttrs[self.smilesAttr]
            titleAttr = self.allAttrs[self.titleAttr]
            smiles = column_data_as_str(self.data, attr)
            titles = column_data_as_str(self.data, titleAttr)
            assert self.thumbnailView.count() == 0
            assert len(self.data) == len(smiles)

            items = [{}] * len(smiles)
            for i, (smiles, title) in enumerate(zip(smiles, titles)):
                if not smiles:  # skip missing
                    continue
                items[i] = {
                    Qt.DisplayRole: title,
                    Qt.EditRole: title,
                    Qt.DecorationRole: smiles,
                    Qt.ToolTipRole: smiles,
                    SmilesDataRole: smiles,
                }
                self.items.append(_ImageItem(i, smiles))

            model = PyListModel([""] * len(items))
            for i, data in enumerate(items):
                model.setItemData(model.index(i), data)
            self.thumbnailView.setModel(model)
            self.thumbnailView.selectionModel().selectionChanged.connect(
                self.onSelectionChanged
            )

    def clearModel(self):
        self.items = []
        model = self.thumbnailView.model()
        if model is not None:
            selmodel = self.thumbnailView.selectionModel()
            selmodel.selectionChanged.disconnect(self.onSelectionChanged)
            model.clear()
        self.thumbnailView.setModel(None)
        self.delegate.deleteLater()
        self.delegate = MoleculeViewDelegate()
        self.thumbnailView.setItemDelegate(self.delegate)

    def updateSize(self):
        self.thumbnailView.setIconSize(QSize(self.imageSize, self.imageSize))

    def updateTitles(self):
        titleAttr = self.allAttrs[self.titleAttr]
        titles = column_data_as_str(self.data, titleAttr)
        model = self.thumbnailView.model()
        for i, item in enumerate(self.items):
            model.setData(model.index(i, 0), titles[item.index], Qt.EditRole)

    @Slot()
    def onSelectionChanged(self):
        items = self.items
        smodel = self.thumbnailView.selectionModel()
        indices = [idx.row() for idx in smodel.selectedRows()]
        self.selectedIndices = [items[i].index for i in indices]
        self.commit()

    def commit(self):
        if self.data:
            if self.selectedIndices:
                selected = self.data[self.selectedIndices]
            else:
                selected = None
            self.Outputs.data.send(selected)
        else:
            self.Outputs.data.send(None)

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()


def column_data_as_str(
        table: Orange.data.Table, var: Orange.data.Variable
) -> Sequence[str]:
    var = table.domain[var]
    data, _ = table.get_column_view(var)
    return list(map(var.str_val, data))
