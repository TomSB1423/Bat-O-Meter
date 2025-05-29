def pytest_addoption(parser):
    parser.addoption(
        "--visualize-tracking",
        action="store_true",
        default=False,
        help="Show OpenCV visualization of tracker performance during tests.",
    )
