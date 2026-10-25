def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the real frozen detectors (needs RUN_PARITY=1, a GPU is fine)")
