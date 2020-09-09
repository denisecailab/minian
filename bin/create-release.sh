#!/bin/bash
#

set -e

VERSION="$1"
if [ "$VERSION" = "" ]; then
    echo >&2 "error: Usage: $0 <VERSION> [<branch>]"
    exit 1
fi

BRANCH=$2
if [ "$#" == 1 ]; then
    BRANCH=master
fi

VERSION_PAT="^v([1-9][0-9]*|0)\.([1-9][0-9]*|0)\.([1-9][0-9]*|0)$"
if ! [[ "$VERSION" =~ $VERSION_PAT ]] ; then
    echo >&2 "error: Illegal version number"
    echo >&2 "format should be 'vX.Y.Z' according to sematic versioning; for example 'v0.1.2'"
    exit 1
fi

if git show-ref --quiet --verify "refs/tags/$VERSION" ; then
    echo >&2 "error: Tag already exists"
    exit 1
fi

if [ "$(git status --porcelain)" != "" ] ; then
    echo >&2 "error: Uncommited changes"
    exit 1
fi

git fetch

HEAD="$(git rev-parse --verify HEAD)"

if [ "$(git show-ref --verify --hash refs/heads/"$BRANCH")" != "$HEAD" ] ; then
    echo >&2 "error: Must be on '$BRANCH'"
    exit 1
fi

if [ "$(git show-ref --verify --hash refs/remotes/origin/"$BRANCH")" != \
        "$HEAD" ] ; then
    echo >&2 "error: Local '$BRANCH' not equal to 'origin/$BRANCH'"
    exit 1
fi

if ! git merge-base --is-ancestor origin/latest-release origin/master ; then
    echo >&2 "error: 'latest-release' must be an ancestor of 'master'"
    echo >&2 "hint: This is required to make sure that all previous releases" \
        "become part of the (git) history of new releases. Use" \
        "'git merge <VERSION>' on 'master' to fix this, or when no actual" \
        "changes are to be merged use 'git merge -s ours <VERSION>'."
    exit 1
fi

if ! git merge-base --is-ancestor origin/latest-release "$BRANCH" ; then
    echo >&2 "error: 'latest-release' must be an ancestor of '$BRANCH'"
    echo >&2 "hint: When creating a hotfix release for 'production' after " \
        "creating a normal release, reset 'latest-release' using " \
        "'git push origin +origin/production:latest-release'"
    exit 1
fi

echo "$VERSION" > VERSION
git add VERSION

if [ -f "setup.py" ]; then
    sed -e "s/version=.*/version='$VERSION',/" setup.py > setup.py.tmp
    mv -- setup.py.tmp setup.py
    git add setup.py
fi

git commit -m "Bump version to $VERSION"

git tag -a "$VERSION" -m "Version $VERSION"

git push --atomic origin "$BRANCH" "$BRANCH":latest-release "$VERSION"

if ! git merge-base --is-ancestor "$BRANCH" origin/master ; then
    echo >&2 "hint: Don't forget to merge '$VERSION' in 'master'"
fi
