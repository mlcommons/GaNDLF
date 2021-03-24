from pytest import fixture


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store"
    )


@fixture()
def device(request):
    return request.config.getoption("--device")