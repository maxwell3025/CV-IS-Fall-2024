import copy
from typing import Any, TypedDict, Generator
import yaml

class MambaOcrConfig(TypedDict):
    dataset_type: str
    dataset_config: dict

    model_type: str
    model_config: dict

    optimizer_type: str
    optimizer_config: dict

    train_config: dict

    val_config: dict

def overwrite(src: Any, target: Any) -> Any:
    """Recursively overwrite the contents of target with src.

    Args:
        src: An object containing the attributes we want to write into target.
        target: An object that we want to overwrite with new data.

    Returns:
        A new object containing all of the attributes of src.
    """
    result = copy.deepcopy(target)
    if type(result) == dict and type(src) == dict:
        for src_key in src:
            result[src_key] = overwrite(src[src_key], result[src_key])
        return result
    elif type(result) == list and type(src) == list:
        for index, value in enumerate(src):
            if index < len(result):
                result[index] = overwrite(value, result[index])
            else:
                result.append(value)
        return result
    else:
        return src
        
def generate_cases(filename: str) -> Generator[MambaOcrConfig, None, None]:
    """TODO write this documentation

    Args:
        filename: _description_

    Yields:
        _description_
    """
    with open(filename, "r") as stream:
        data = yaml.safe_load(stream)
    base = data["base"]

    sweep_config = data["cases"]
    assert type(sweep_config) == list
    for version in sweep_config:
        yield overwrite(version, base)
