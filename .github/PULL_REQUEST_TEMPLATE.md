<!-- Replace ISSUE_NUMBER with the issue that will be auto-linked to close after merging this PR -->
Fixes #ISSUE_NUMBER

## Proposed Changes
<!-- Bullet pointed list of changes, please try to keep code changes as small as possible-->
- 

## Checklist

<!-- You do not need to complete all the items by the time you submit the pull request, 
but PRs are more likely to be merged quickly if all the tasks are done. -->

<!-- Replace `[ ]` with `[x]` in all the boxes that apply.
Note that if a box is left unchecked, PR merges will take longer than usual.
-->
- [ ] I have read the [`CONTRIBUTING`](https://github.com/mlcommons/GaNDLF/blob/master/CONTRIBUTING.md) guide.
- [ ] My PR is based from the [current GaNDLF master ](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/keeping-your-local-repository-in-sync-with-github/syncing-your-branch-in-github-desktop?platform=windows).
- [ ] Non-breaking change (does **not** break existing functionality): provide **as many** details as possible for _any_ breaking change.
- [ ] Function/class source code documentation added/updated.
- [ ] Code has been [blacked](https://github.com/psf/black#usage) for style consistency.
- [ ] If applicable, version information [has been updated in GANDLF/version.py](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/version.py).
- [ ] If adding a git submodule, add to list of exceptions for black styling in [pyproject.toml](https://github.com/mlcommons/GaNDLF/blob/master/pyproject.toml) file.
- [ ] [Usage documentation](https://github.com/mlcommons/GaNDLF/blob/master/docs) has been updated, if appropriate.
- [ ] Tests added or modified to [cover the changes](https://app.codecov.io/gh/mlcommons/GaNDLF); if coverage is reduced, please give explanation.
- [ ] If customized dependency installation is required (i.e., a separate `pip install` step is needed for PR to be functional), please ensure it is reflected in all the files that control the CI, namely: [python-test.yml](https://github.com/mlcommons/GaNDLF/blob/master/.github/workflows/python-test.yml), and all docker files [[1](https://github.com/mlcommons/GaNDLF/blob/master/Dockerfile-CPU),[2](https://github.com/mlcommons/GaNDLF/blob/devcontainer_build_fix/Dockerfile-CUDA11.6),[3](https://github.com/mlcommons/GaNDLF/blob/master/Dockerfile-ROCm)].
