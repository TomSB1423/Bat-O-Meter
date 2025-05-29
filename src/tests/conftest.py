def pytest_addoption(parser):
    parser.addoption(
        "--debug-frames",
        action="store_true",
        default=False,
        help="Show frame visualization during tests.",
    )
