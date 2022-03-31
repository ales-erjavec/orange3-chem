import logging
from Orange.misc.utils.embedder_utils import EmbedderCache
from concurrent.futures import CancelledError
from typing import overload, Sequence, Optional, Callable, Union, Tuple
import numpy as np


from Orange.data import ContinuousVariable, Domain, Table, StringVariable
from Orange.misc.server_embedder import ServerEmbedderCommunicator

log = logging.getLogger(__name__)


MODELS = {
    'smiles': {
        'name': 'CNN-Based SMILES Embedder',
        'description': 'CNN model trained on Pharmacologic Action MeSH terms classification',
        'target_smiles_length': 1021,
        'batch_size': 500,  # ??
        'is_local': False,
        'layers': ['penultimate'],
        'order': 0
    },
}


class ServerEmbedder(ServerEmbedderCommunicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_type = 'text/plain'

    async def _encode_data_instance(self, data_instance: str) -> bytes:
        """
        This is just an implementation for test purposes. We just return
        a sample bytes which is id encoded to bytes.
        """
        return data_instance.encode('utf-8')


class EmbeddingCancelledException(CancelledError):
    """Thrown when the embedding task is cancelled from another thread.
    (i.e. MoleculeEmbedder.cancelled attribute is set to True).
    """


class MoleculeEmbedder:
    """
    Client side functionality for accessing a remote molecule embedding
    backend.
    """
    _embedder = None

    def __init__(self, model="smiles", server_url='https://api.garaza.io/'):
        self.model = model
        self.server_url = server_url
        self._model_settings = self._get_model_settings_confidently()

    def _get_model_settings_confidently(self):
        if self.model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ', '.join(MODELS.keys())
            raise ValueError(model_error.format(self.model, available_models))

        return MODELS[self.model]

    def is_local_embedder(self) -> bool:
        """
        Tells whether selected embedder is local or not.
        """
        return self._model_settings.get("is_local", False)

    def _init_embedder(self) -> None:
        """
        Init local or server embedder.
        """
        if self.is_local_embedder():
            self._embedder = LocalEmbedder(self.model, self._model_settings)
        else:
            self._embedder = ServerEmbedder(
                self.model,
                self._model_settings["batch_size"],
                self.server_url,
                "chem",
            )

    @overload
    def __call__(self, /, smiles: Sequence[str], *, callback: Optional[Callable] = None) -> Sequence[str]: ...
    @overload
    def __call__(self, /, data: Table, col: Union[str, StringVariable], *, callback: Optional[Callable] = None) -> Table: ...

    def __call__(self, *args, **kwargs):
        if len(args) and isinstance(args[0], Table) or \
                ("data" in kwargs and isinstance(kwargs["data"], Table)):
            return self.from_table(*args, **kwargs)
        elif (len(args) and isinstance(args[0], (np.ndarray, list))) or \
                ("smiles" in kwargs and isinstance(kwargs["smiles"], (np.ndarray, list))):
            return self.from_smiles(*args, **kwargs)
        else:
            raise TypeError

    def from_table(self, data: Table, col="SMILES", callback=None):
        smiles = data[:, col].metas.flatten()
        embeddings = self.from_smiles(smiles, callback)
        return MoleculeEmbedder.prepare_output_data(data, embeddings)

    def from_smiles(self, smiles: Sequence[str], callback=None):
        """Send the smiles to the remote server in batches. The batch size
        parameter is set by the http2 remote peer (i.e. the server).

        Parameters
        ----------
        smiles: list
            A list of smiles for moelcules to be embedded.

        callback: callable (default=None)
            A function that is called after each smiles is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the smiles.

        Returns
        -------
        embeddings: array-like
            Array-like of float16 arrays (embeddings) for
            successfully embedded smiles and Nones for skipped smiles.

        Raises
        ------
        ConnectionError:
            If disconnected or connection with the server is lost
            during the embedding process.

        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        self._init_embedder()
        return self._embedder.embedd_data(smiles, processed_callback=callback)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.set_canceled()

    def __del__(self) -> None:
        self.__exit__(None, None, None)

    @staticmethod
    def construct_output_data_table(embedded_smiles, embeddings):
        X = np.hstack((embedded_smiles.X, embeddings))
        Y = embedded_smiles.Y

        attributes = [ContinuousVariable.make('n{:d}'.format(d))
                      for d in range(embeddings.shape[1])]
        attributes = list(embedded_smiles.domain.attributes) + attributes

        domain = Domain(
            attributes=attributes,
            class_vars=embedded_smiles.domain.class_vars,
            metas=embedded_smiles.domain.metas
        )

        return Table(domain, X, Y, embedded_smiles.metas)

    @staticmethod
    def prepare_output_data(
            input_data,
            embeddings: Sequence[Optional[Sequence[float]]]
    ) -> Tuple[Table, Table, int]:
        skipped_smiles_bool = np.array([x is None for x in embeddings])

        if np.any(skipped_smiles_bool):
            skipped_smiles = input_data[skipped_smiles_bool]
            skipped_smiles = Table(skipped_smiles)
            skipped_smiles.ids = input_data.ids[skipped_smiles_bool]
            num_skipped = len(skipped_smiles)
        else:
            num_skipped = 0
            skipped_smiles = None

        embedded_smiles_bool = np.logical_not(skipped_smiles_bool)

        if np.any(embedded_smiles_bool):
            embedded_smiles = input_data[embedded_smiles_bool]

            embeddings = [
                e for e, b in zip(embeddings, embedded_smiles_bool) if b
            ]
            embeddings = np.vstack(embeddings)

            embedded_smiles = MoleculeEmbedder.construct_output_data_table(
                embedded_smiles,
                embeddings
            )
            embedded_smiles.ids = input_data.ids[embedded_smiles_bool]
        else:
            embedded_smiles = None

        return embedded_smiles, skipped_smiles, num_skipped

    @staticmethod
    def filter_string_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.is_string]

    def clear_cache(self) -> None:
        """
        Function clear cache for the selected embedder. If embedder is loaded
        cache is cleaned from its dict otherwise we load cache and clean it
        from file.
        """
        if self._embedder:
            # embedder is loaded so we clean its cache
            self._embedder.clear_cache()
        else:
            # embedder is not initialized yet - clear it cache from file
            cache = EmbedderCache(self.model)
            cache.clear_cache()

    def set_canceled(self) -> None:
        """
        Cancel the embedding
        """
        if self._embedder is not None:
            self._embedder.set_cancelled()


if __name__ == "__main__":
    em = ServerEmbedder(
        model_name="smiles",
        max_parallel_requests=500,
        server_url="https://api.garaza.io/",
        embedder_type="chem"
    )
    print(em.embedd_data([
        "C(=O)[C@H](CCCCNC(=O)OCCOC)NC(=O)OCCOC",
        "OCC",
        "CCCCNC(=O)OCNC(=O)OCCOC",
        "H2O"
    ]))