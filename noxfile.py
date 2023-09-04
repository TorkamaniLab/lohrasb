import nox

nox.options.sessions = [
    "install_requirements",
    "test_optimize_by_tune",
    "test_optimize_by_tunegridsearchcv",
    "test_optimize_by_optunasearch",
    "test_optimize_by_optunasearchcv",
    "test_optimize_by_tunerandomsearchcv",
    "test_optimize_by_gridsearchcv",
    "test_optimize_by_randomsearchcv",
    "lint_lohrasb"
]


@nox.session
def lint_lohrasb(session):
    """Run lint session using nox"""
    # Install linters
    session.install("black", "isort")
    # Run isort and black
    session.run("isort", "./lohrasb/")
    session.run("black", "./lohrasb/")

# This function is to install requirements only once and will be reused by others.
@nox.session(reuse_venv=True)
def install_requirements(session):
    """Install all test requirements."""
    session.install("-r", "requirements_test.txt")


@nox.session(reuse_venv=True)
def test_optimize_by_tune(session):
    """Run tests for optimize_by_tune."""
    session.run("pytest", "tests/test_optimize_by_tune.py")


@nox.session(reuse_venv=True)
def test_optimize_by_tunegridsearchcv(session):
    """Run tests for optimize_by_tunegridsearchcv."""
    session.run("pytest", "tests/test_optimize_by_tunegridsearchcv.py")


@nox.session(reuse_venv=True)
def test_optimize_by_optunasearch(session):
    """Run tests for optimize_by_optunasearch."""
    session.run("pytest", "tests/test_optimize_by_optunasearch.py")


@nox.session(reuse_venv=True)
def test_optimize_by_optunasearchcv(session):
    """Run tests for optimize_by_optunasearchcv."""
    session.run("pytest", "tests/test_optimize_by_optunasearchcv.py")


@nox.session(reuse_venv=True)
def test_optimize_by_tunerandomsearchcv(session):
    """Run tests for optimize_by_tunerandomsearchcv."""
    session.run("pytest", "tests/test_optimize_by_tunerandomsearchcv.py")


@nox.session(reuse_venv=True)
def test_optimize_by_gridsearchcv(session):
    """Run tests for optimize_by_gridsearchcv."""
    session.run("pytest", "tests/test_optimize_by_gridsearchcv.py")


@nox.session(reuse_venv=True)
def test_optimize_by_randomsearchcv(session):
    """Run tests for optimize_by_randomsearchcv."""
    session.run("pytest", "tests/test_optimize_by_randomsearchcv.py")
