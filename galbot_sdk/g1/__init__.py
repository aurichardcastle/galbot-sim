"""galbot_sdk.g1 — G1 platform wrappers (GalbotSim).

Verbatim port of the official ``g1/__init__.py`` wrapper pattern
(api_contract.md §1), pre-bound to ``MachineType.G1``: re-exports the entire
base-module namespace via ``from .. import *`` and defines three no-arg
singleton wrapper classes that delegate every attribute access to the base
singletons obtained through ``get_instance(MachineType.G1)``.

Documented lifecycle (identical to the official docstring contract)::

    robot = GalbotRobot()
    robot.init()
    ... use the SDK ...
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()            # REQUIRED on exit

The wrappers are singletons: ``GalbotRobot()`` always returns the same
object; ``init()`` is effective only once; after ``destroy()`` the SDK cannot
be re-initialized in the same process. ``isinstance`` checks against the base
classes fail on the wrappers — exactly like the real SDK (§10.1).
"""

from .. import MachineType
from .. import GalbotNavigation as _BaseGalbotNavigation
from .. import GalbotMotion as _BaseGalbotMotion
from .. import GalbotRobot as _BaseGalbotRobot
from .. import *  # noqa: F401,F403 — full base-module namespace re-export


class GalbotRobot:
    """No-arg singleton wrapper over the base GalbotRobot, bound to G1."""

    _instance = None      # base (pybind-lookalike) singleton
    _py_instance = None   # python wrapper singleton

    def __new__(cls):
        if cls._py_instance is None:
            cls._py_instance = super().__new__(cls)
        return cls._py_instance

    def __init__(self):
        if GalbotRobot._instance is None:
            GalbotRobot._instance = _BaseGalbotRobot.get_instance(MachineType.G1)
        self._impl = GalbotRobot._instance

    def __getattr__(self, name):
        if name == "_impl":  # guard: never delegate the delegate itself
            raise AttributeError(name)
        return getattr(self._impl, name)


class GalbotMotion:
    """No-arg singleton wrapper over the base GalbotMotion, bound to G1."""

    _instance = None
    _py_instance = None

    def __new__(cls):
        if cls._py_instance is None:
            cls._py_instance = super().__new__(cls)
        return cls._py_instance

    def __init__(self):
        if GalbotMotion._instance is None:
            GalbotMotion._instance = _BaseGalbotMotion.get_instance(MachineType.G1)
        self._impl = GalbotMotion._instance

    def __getattr__(self, name):
        if name == "_impl":
            raise AttributeError(name)
        return getattr(self._impl, name)


class GalbotNavigation:
    """No-arg singleton wrapper over the base GalbotNavigation, bound to G1."""

    _instance = None
    _py_instance = None

    def __new__(cls):
        if cls._py_instance is None:
            cls._py_instance = super().__new__(cls)
        return cls._py_instance

    def __init__(self):
        if GalbotNavigation._instance is None:
            GalbotNavigation._instance = _BaseGalbotNavigation.get_instance(MachineType.G1)
        self._impl = GalbotNavigation._instance

    def __getattr__(self, name):
        if name == "_impl":
            raise AttributeError(name)
        return getattr(self._impl, name)


# __all__ = every public global (wrapper names shadow the base classes;
# GalbotPerception reaches this namespace via the star re-export).
__all__ = [_n for _n in list(globals()) if not _n.startswith("_")]
