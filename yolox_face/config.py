import importlib.util
from pathlib import Path


def load_config(path: str):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CFG