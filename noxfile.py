import argparse
import nox

nox.options.sessions = ["tests_lohrasb", "lint_lohrasb", "release_lohrasb"]

SESSIONS = ["tests", "lint", "release"]


@nox.session(python=False)
def tests_lohrasb(session):
    """Run test session using nox"""
    session.run("poetry", "shell")
    session.run("poetry", "install")
    session.run("pytest")


@nox.session
def lint_lohrasb(session):
    """Run lint session using nox"""
    session.install("flake8", "black", "isort")
    session.run("isort", "./lohrasb/")
    session.run("black", "./lohrasb/")
    session.run(
        "flake8", "--ignore=E501,I202,F811,W503,E203,F401,F401,F405,F403,F841", "./lohrasb/"
    )


@nox.session
def release_lohrasb(session):
    """
    Kicks off an automated release process by creating and pushing a new tag.
    Invokes bump2version with the posarg setting the version.
    Usage:
    $ nox -s release -- [major|minor|patch]
    """
    parser = argparse.ArgumentParser(description="Release a semver version.")
    parser.add_argument(
        "version",
        type=str,
        nargs=1,
        choices={"major", "minor", "patch"},
        help="The type of semver release to make.",
    )
    parser.add_argument("username", type=str, nargs=1, help="Username for git")
    parser.add_argument("useremail", type=str, nargs=1, help="User email for git")
    parser.add_argument("gitpassword", type=str, nargs=1, help="Git password for git")

    args: argparse.Namespace = parser.parse_args(args=session.posargs)
    version, username, useremail, gitpassword = map(
        lambda x: x.pop(),
        [args.version, args.username, args.useremail, args.gitpassword],
    )

    session.install("bump2version")

    session.log(f"Bumping the {version!r} version")
    session.run("bump2version", "--allow-dirty", version)
    session.log("Pushing the new tag")
    session.run("git", "config", "--global", "user.email", useremail, external=True)
    session.run("git", "config", "--global", "user.name", username, external=True)
    session.run(
        "git", "config", "--global", "user.password", gitpassword, external=True
    )
    session.run(
        "git",
        "remote",
        "set-url",
        "origin",
        f"https://{username}:{gitpassword}@github.com/TorkamaniLab/lohrasb.git",
        external=True,
    )
    session.run("git", "branch", "temp-branch", external=True)
    session.run("git", "checkout", "main", external=True)
    session.run("git", "merge", "temp-branch", external=True)
    session.run("git", "branch", "--delete", "temp-branch", external=True)
    session.run("git", "push", "origin", external=True)
    session.run("git", "push", "--tags", external=True)
