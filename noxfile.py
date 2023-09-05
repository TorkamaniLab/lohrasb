import nox

nox.options.sessions = ["tests_lohrasb", "lint_lohrasb"]
test_files = [
    "tests/test_optimize_by_tune.py",
    "tests/test_optimize_by_tunegridsearchcv.py",
    "tests/test_optimize_by_optunasearch.py",
    "tests/test_optimize_by_optunasearchcv.py",
    "tests/test_optimize_by_tunerandomsearchcv.py",
    "tests/test_optimize_by_gridsearchcv.py",
    "tests/test_optimize_by_randomsearchcv.py",
]
@nox.session  # Removed python=PYTHON_VERSIONS
def tests_lohrasb(session):
    # The list of test files to be executed
    """Install test dependencies."""
    session.install("-r", "requirements_test.txt")
    session.run("pytest", test_files[0])
    session.run("pytest", test_files[1])
    session.run("pytest", test_files[2])
    session.run("pytest", test_files[3])
    session.run("pytest", test_files[4])
    session.run("pytest", test_files[5])
    session.run("pytest", test_files[6])

@nox.session
def lint_lohrasb(session):
    """Run lint session using nox"""
    # Install linters
    session.install("black", "isort")
    # Run isort and black
    session.run("isort", "./lohrasb/")
    session.run("black", "./lohrasb/")