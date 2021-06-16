from pytest import fixture


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="cpu", help="device option: cpu or cuda"
    )


@fixture()
def device(request):
    return request.config.getoption("--device")
