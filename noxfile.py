import nox

nox.options.sessions = ["tests_lohrasb", "lint_lohrasb"]
PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]

@nox.session(python=PYTHON_VERSIONS)
def tests_lohrasb(session):
    """Run test session using nox"""
    # Install test dependencies from requirements_test.txt
    session.install("-r", "requirements_test.txt")
    # Run pytest
    session.run("pytest")

@nox.session
def lint_lohrasb(session):
    """Run lint session using nox"""
    # Install linters
    session.install("black", "isort")
    # Run isort and black
    session.run("isort", "./lohrasb/")
    session.run("black", "./lohrasb/")
