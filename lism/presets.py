"""Build a Guide: one cell of the {Point, Iso, Fit} x {Id, Cov, JVP, VJP, FD, Path} grid."""
from .lism import Guide
from .weighting import DEFAULT_L, WEIGHTINGS
from .response import RESPONSES


def guide(weight, respond, cov=None, alpha=1., L=None, window=(.01, .995), cap=None):
    """One cell of the grid, e.g. guide("fit", "path", cov). Point pairs with the logistic L."""
    if weight not in WEIGHTINGS or respond not in RESPONSES:
        raise KeyError(f"Weighting is one of {sorted(WEIGHTINGS)}, response one of {sorted(RESPONSES)}")
    if weight != "point" and cov is None:
        raise ValueError("Iso and Fit weighting need a fitted covariance")
    if respond == "cov" and cov is None:
        raise ValueError("The Cov response needs a fitted covariance")
    used = cov.iso() if weight == "iso" else cov          # the response shares the weighting's covariance
    return Guide(L or DEFAULT_L[weight](), WEIGHTINGS[weight](cov), RESPONSES[respond](used), alpha, window, cap)


def preset(weight, respond, task):
    """A grid cell with the strength, window and cap that were tuned for this task's probe.

    They are stored with the probe (config.json, "presets"): a strength means nothing for another probe.
    """
    tuned = task.config.get("presets", {}).get(f"{weight}_{respond}")
    if tuned is None:
        raise KeyError(f"{weight} x {respond} has no tuned strength for this task: use guide() and pass alpha")
    return guide(weight, respond, task.cov, tuned["alpha"], window=tuple(tuned.get("window", (.01, .995))), cap=tuned.get("cap"))
