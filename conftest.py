# Empty on purpose.
#
# pytest adds the directory containing the root conftest.py to sys.path, which
# is what lets tests do `from preprocessing.manifest import ...` the same way
# scripts run from the repo root do. Without this file, pytest would only put
# tests/ on the path and those imports would fail.
