import pkgutil
from typing import Sequence, TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from tensorflow.lite import Interpreter

# SMILES vocabulary
vocabulary = {
    'c': 53,
    'h': 58,
    '(': 4,
    'U': 42,
    'a': 51,
    'y': 71,
    'g': 57,
    'E': 28,
    'B': 25,
    'H': 31,
    'D': 27,
    'f': 56,
    'o': 64,
    '9': 20,
    '.': 9,
    'm': 62,
    '5': 16,
    'W': 44,
    'Y': 46,
    '0': 11,
    'k': 60,
    ']': 50,
    'N': 36,
    'e': 55,
    't': 68,
    '6': 17,
    '4': 15,
    'l': 61,
    'O': 37,
    'A': 24,
    'I': 32,
    'M': 35,
    'F': 29,
    'K': 33,
    ':': 21,
    '8': 19,
    'P': 38,
    '#': 1,
    'S': 40,
    'G': 30,
    '=': 22,
    'r': 66,
    'n': 63,
    'Z': 47,
    '@': 23,
    '7': 18,
    '+': 7,
    'C': 26,
    'R': 39,
    '1': 12,
    'p': 65,
    '*': 6,
    'b': 52,
    '$': 2,
    ')': 5,
    'X': 45,
    '%': 3,
    '3': 14,
    'T': 41,
    '/': 10,
    '[': 48,
    's': 67,
    'd': 54,
    'i': 59,
    'L': 34,
    '2': 13,
    '-': 8,
    'V': 43,
    'v': 70,
    '\\': 49,
    'u': 69
}


def post_pad_sequence(seq: Sequence, length: int, value=0) -> Sequence:
    slen = len(seq)
    padlen = length - slen
    seq = [*seq, *([value] * padlen)]
    return seq


def get_cnn_fingerprint(
        smiles: Sequence[str],
        model: Optional["Interpreter"] = None
) -> np.ndarray:
    if model is None:
        import tensorflow as tf
        content = pkgutil.get_data(__name__, "data/smiles-cnn-embedder.tflite")
        model = tf.lite.Interpreter(model_content=content)
    input_d = model.get_input_details()[0]
    output_d = model.get_output_details()[0]
    shape = input_d["shape"]
    seqs = [[vocabulary[c] for c in list(s)] for s in smiles]

    data = [post_pad_sequence(seq, shape[1]) for seq in seqs]
    data = np.array(data, input_d["dtype"])
    model.resize_tensor_input(input_d["index"], (len(data), *shape[1:]))
    model.allocate_tensors()
    model.set_tensor(input_d["index"], data)
    model.invoke()
    return model.get_tensor(output_d["index"])


class MissingOptionalDependencyError(ImportError):
    pass


class LocalEmbedder:

    embedder = None

    def __init__(self, model, model_settings):
        self.model = model
        self._load_model()
        self._cancelled = False

    def _load_model(self):
        try:
            import tensorflow as tf
        except ImportError as ex:
            raise MissingOptionalDependencyError(
                "Local embedder requires tensorflow package to run. "
                "Please install it and try again."
            ) from ex
        content = pkgutil.get_data(__name__, "data/smiles-cnn-embedder.tflite")
        self.model = tf.lite.Interpreter(model_content=content)

    def embedd_data(self, smiles: Sequence[str], processed_callback=None):
        return get_cnn_fingerprint(smiles, self.model)

    def set_cancelled(self):
        self._cancelled = True

