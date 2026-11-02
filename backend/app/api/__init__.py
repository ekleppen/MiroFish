"""
API routes module
"""

from flask import Blueprint

graph_bp = Blueprint('graph', __name__)
simulation_bp = Blueprint('simulation', __name__)
report_bp = Blueprint('report', __name__)
population_bp = Blueprint('population', __name__)
experiments_bp = Blueprint('experiments', __name__)

from . import graph  # noqa: E402, F401
from . import simulation  # noqa: E402, F401
from . import report  # noqa: E402, F401
from . import population  # noqa: E402, F401
from . import experiments  # noqa: E402, F401

