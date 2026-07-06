# @File (label="Select Root Directory", style="directory") rootDir
# @File (label="Select Pipeline Output Root", style="directory") outputRoot
# @String (label="File Extension", value=".oir") ext
# @Integer (label="Max Threads", value=8) threads
# @String (label="Existing output mode", value="skip") existingMode
# @String (label="Projection mode", value="auto") projectionMode
# @String (label="Metadata mode", value="update-missing") metadataMode
# @String (label="Stim analog export mode", value="projected") stimExportMode
#
# ASCII-only wrapper for Fiji/Jython headless execution.
#
# Some Fiji/Jython combinations choke on non-ASCII source text when opening the
# worker script directly. This shim stays ASCII-only, reads the real worker as
# UTF-8, and executes it from an in-memory string instead.

from __future__ import print_function

import os
import sys

from java.io import File

WRAPPER_PARAM_NAMES = (
    "rootDir",
    "outputRoot",
    "ext",
    "threads",
    "existingMode",
    "projectionMode",
    "metadataMode",
    "stimExportMode",
)


def _load_utf8_worker_source():
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    worker_path = os.path.join(wrapper_dir, "01_fiji_totif_worker.py")
    handle = open(worker_path, "rb")
    try:
        source = handle.read().decode("utf-8")
    finally:
        handle.close()
    if source.startswith(u"\ufeff"):
        source = source[1:]
    source = u"from __future__ import unicode_literals\n" + source
    return worker_path, source


def _split_key_value_items(raw_text):
    items = []
    current = []
    in_quotes = False
    escape = False
    for ch in raw_text:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            current.append(ch)
            escape = True
            continue
        if ch == '"':
            current.append(ch)
            in_quotes = not in_quotes
            continue
        if ch == "," and not in_quotes:
            piece = "".join(current).strip()
            if piece:
                items.append(piece)
            current = []
            continue
        current.append(ch)
    piece = "".join(current).strip()
    if piece:
        items.append(piece)
    return items


def _decode_scalar(raw_value):
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    value = value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def _coerce_wrapper_param(key, value):
    if key in ("rootDir", "outputRoot"):
        return File(value)
    if key == "threads":
        try:
            return int(value)
        except Exception:
            return 1
    return value


def _parse_cli_args():
    if len(sys.argv) < 2:
        return {}
    raw_text = sys.argv[1]
    if not raw_text or "=" not in raw_text:
        return {}

    parsed = {}
    for item in _split_key_value_items(raw_text):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = _coerce_wrapper_param(key, _decode_scalar(raw_value))
    return parsed


def _execute_worker():
    worker_path, source = _load_utf8_worker_source()
    globals_dict = {
        "__file__": worker_path,
        "__name__": "__main__",
    }
    for name in WRAPPER_PARAM_NAMES:
        if name in globals():
            globals_dict[name] = globals()[name]
    globals_dict.update(_parse_cli_args())
    exec(compile(source, worker_path, "exec"), globals_dict, globals_dict)


_execute_worker()
