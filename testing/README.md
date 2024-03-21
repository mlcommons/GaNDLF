# Running tests locally

## Prerequisites

Some additional steps are required for running tests.

Firstly, install optional dependencies (if still not):

```shell
pip install mlcube_docker openvino==2023.0.1
```

Second, download and unpack the following testing data to `testing/` subfolder:

```shell
# it's assumed you are in `GaNDLF/` repo root directory
cd testing/
wget https://upenn.box.com/shared/static/y8162xkq1zz5555ye3pwadry2m2e39bs.zip
unzip $FILENAME
```
 
## Running tests

There are two types of tests: unit tests for GaNDLF code, and integration tests for deploying and running mlcubes.

### Unit tests

To run unit tests, use

```shell
# it's assumed you are in `GaNDLF/` repo root directory
pytest . --cov=.
```

So, after tests are run you may explore its coverage report manually. NB: tests may take some time to be passed.

### Integration tests

All integration tests are combined to one script:

```shell
# it's assumed you are in `GaNDLF/` repo root directory
cd testing/
./test_deploy.sh
```
