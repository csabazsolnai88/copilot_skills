# rnd-template
Template repository for RND

Get started by:

1. Adjust the project name in pyproject.toml.
2. Rename the folder ```rnd_template``` to the new project name
3. Generate a lock file (```poetry lock```)
4. Change ```rnd_template``` to your project name in .github/workflows/test.yml and .github/workflows/lint.yml
5. Rewrite this ```README.md```

The template includes:
* A minimal pyproject.toml with `ruff` and `pre-commit`
* Github runners
* Pre-commit config `ruff` and a few other checks such as secrets and large files
* Dependabot for dependency checking

### For a setup with venv and pre-commit hooks:
```bash
python3 -m venv --prompt $(basename $PWD) venv
source venv/bin/activate

pip install -U pip
pip install poetry  # good to fix to a recent version of poetry

export POETRY_CACHE_DIR=/mnt/play/$USER/.cache # So the home directory doesn't explode

poetry install --no-cache
pre-commit install
```

## Coding style
General:
- Use RND template repo to start a new project
- Use ruff (previously black) before pushing
- Commit frequently
- Open a work in progress PR making sure your changes will be merged rather than forgotten
- Review the code yourself before asking for a review (CI/CD run with success)
- Merge fast
- Create separate and small PRs for separate issues (especially after development has stabilised)
- Scout's rule: “Always leave the code cleaner than you found it”

Code Agents:
- Make use of code agents (company licence)
- Concise code and fewer lines is better code
- Don't waste your reviewers time with Agent trash (code, comments)

Code structure:
- Don't repeat yourself (DRY)
- Modularize with functions, classes and submodules
- Hard code as little as possible, use CLI arguments and constants
- Write unittests
- Use dataclasses instead of tuples

Documentation:
- Add type hints
- Optionally add docstrings: only explanation (not parameters) and output
- Comment code if it make sense
- Document use in README

Packages:
- Use polars instead of pandas
- Use pathlib instead of os and str


## Ruff linting rules

All rules are documented [here](https://docs.astral.sh/ruff/rules/)

If a Ruff linting rule is preventing you from committing

**Workaround 1**

`pyproject.toml`: IGNORE = ["E501"]

**Workaround 2**

`.py` offending line:

> `some python code  #noqa`
>
> or 
>
> `some python code  #noqa: E501`

**Workaround 3**

`git commit --no-verify`

**Workaround 4**

remove rule from SELECT = [...] in `pyproject.toml`
