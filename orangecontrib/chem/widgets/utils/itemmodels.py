from typing import Any, Callable, Mapping, Optional, Union

from AnyQt.QtCore import QAbstractListModel, QModelIndex, Qt


class ListModelAdapter(QAbstractListModel):
    __slots__ = ("__size", "__data", "__data_setters")

    # data getter
    Dispatch = Callable[[int], Any]
    # data setter dispatch
    SetterDispatch = Callable[[int, Any], Any]

    def __init__(self, size, dispatch: Mapping[int, Dispatch], **kwargs) -> None:
        self.__size = size
        self.__data = dict(dispatch)
        self.__data_setters = {}
        super().__init__(**kwargs)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        else:
            return self.__size

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row, column = index.row(), index.column()
        N = self.__size
        if not 0 <= row < N and column != 0:
            return None

        delegate = self.dataDelegateForRole(role)
        if delegate is not None:
            return delegate(row)
        else:
            return None

    def setData(
            self, index: QModelIndex, value: Any, role: int = Qt.EditRole
    ) -> bool:
        if not index.isValid():
            return False
        row, column = index.row(), index.column()
        N = self.__size
        if not 0 <= row < N and column != 0:
            return False

        delegate = self.dataSetterDelegateForRole(role)
        if delegate is not None:
            delegate(row, value)
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, 0), (role,)
            )
            return True
        else:
            return False

    def dataDelegateForRole(
            self, role: Union[int, Qt.ItemDataRole]
    ) -> Optional[Callable[[int], Any]]:
        return self.__data.get(role, None)

    def dataSetterDelegateForRole(
            self, role: Union[int, Qt.ItemDataRole]
    ) -> Optional[Callable[[int, Any], Any]]:
        return self.__data_setters.get(role, None)

    def setDelegateForRole(
            self, role: int, delegate: Optional[Dispatch]
    ) -> None:
        if delegate is None:
            self.__data.pop(role)
        else:
            self.__data[role] = delegate
        if self.__size:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.__size - 1, 0),
                (role,),
            )

    def setSetterDelegateForRole(
            self, role, delegate: Optional[SetterDispatch]
    ) -> None:
        if delegate is None:
            self.__data_setters.pop(role)
        else:
            self.__data_setters[role] = delegate
