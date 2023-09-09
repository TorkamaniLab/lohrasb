import nox

# Default sessions to run when no sessions are specified.
nox.options.sessions = ["tests_lohrasb", "lint_lohrasb"]

# List of test files to be executed.
test_files = [
    # TODO need to test more to see why it has problem in windows
    "tests/test_optimize_by_tune.py",
    "tests/test_optimize_by_tunegridsearchcv.py",
    "tests/test_optimize_by_optunasearch.py",
    "tests/test_optimize_by_optunasearchcv.py",
    "tests/test_optimize_by_tunerandomsearchcv.py",
    "tests/test_optimize_by_gridsearchcv.py",
    "tests/test_optimize_by_randomsearchcv.py",
]

@nox.session
def tests_lohrasb(session):
    """Install test dependencies and run pytest for all test files."""
    session.install("-r", "requirements_test.txt")
    
    for test_file in test_files:
        session.run("pytest", test_file)

@nox.session
def lint_lohrasb(session):
    """Run lint session using nox."""
    # Install linters.
    session.install("black", "isort")
    
    # Run isort and black.
    session.run("isort", "./lohrasb/")
    session.run("black", "./lohrasb/")
