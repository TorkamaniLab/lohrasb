import nox

# The list of test files to be executed
test_files = [
    "tests/test_optimize_by_tune.py",
    "tests/test_optimize_by_tunegridsearchcv.py",
    "tests/test_optimize_by_optunasearch.py",
    "tests/test_optimize_by_optunasearchcv.py",
    "tests/test_optimize_by_tunerandomsearchcv.py",
    "tests/test_optimize_by_gridsearchcv.py",
    "tests/test_optimize_by_randomsearchcv.py",
]

# Install requirements
@nox.session(reuse_venv=True)
def install_requirements(session):
    """Install test dependencies."""
    session.install("-r", "requirements_test.txt")

# Session for each test file
@nox.session(reuse_venv=True)
def test_optimize_by_tune(session):
    """Run pytest for test_optimize_by_tune.py."""
    session.run("pytest", test_files[0])

@nox.session(reuse_venv=True)
def test_optimize_by_tunegridsearchcv(session):
    """Run pytest for test_optimize_by_tunegridsearchcv.py."""
    session.run("pytest", test_files[1])

@nox.session(reuse_venv=True)
def test_optimize_by_optunasearch(session):
    """Run pytest for test_optimize_by_optunasearch.py."""
    session.run("pytest", test_files[2])

@nox.session(reuse_venv=True)
def test_optimize_by_optunasearchcv(session):
    """Run pytest for test_optimize_by_optunasearchcv.py."""
    session.run("pytest", test_files[3])

@nox.session(reuse_venv=True)
def test_optimize_by_tunerandomsearchcv(session):
    """Run pytest for test_optimize_by_tunerandomsearchcv.py."""
    session.run("pytest", test_files[4])

@nox.session(reuse_venv=True)
def test_optimize_by_gridsearchcv(session):
    """Run pytest for test_optimize_by_gridsearchcv.py."""
    session.run("pytest", test_files[5])

@nox.session(reuse_venv=True)
def test_optimize_by_randomsearchcv(session):
    """Run pytest for test_optimize_by_randomsearchcv.py."""
    session.run("pytest", test_files[6])
