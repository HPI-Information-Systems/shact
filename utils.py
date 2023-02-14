import inspect
from typing import Callable, Dict

def filter_kwargs(f:Callable, kwargs: Dict) -> Dict:
    argspec = inspect.getfullargspec(f)
    return {k: v for k, v in kwargs.items() if k in argspec.args}