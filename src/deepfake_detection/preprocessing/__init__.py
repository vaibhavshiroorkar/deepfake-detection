"""Step-by-step preprocessing ops shared by the pipeline and the dashboard.

`deepfake_detection.views` owns the cached, contract-checked preprocessing that
training and evaluation run. This package holds the same steps as small pure
functions so the teaching dashboard can show one step at a time.
"""
