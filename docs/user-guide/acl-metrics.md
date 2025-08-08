# ROSE Standard and Custom Metrics

ROSE supports different Machine Learning (ML) Metrics such as `RMSE`, `MAE`, and `F2score` and many more.


## Standard Metrics
For a full list of the supported metrics please refer to the following link [ROSE Standard Metrics](https://github.com/radical-cybertools/ROSE/blob/feature/al_algo_selector/rose/metrics.py)


## Custom Metrics
ROSE allows the user to define additional metrics if not supported by default. To define
a custom metric, you can do the following:

import the operator for the custom metric:
```python
from rose.metrics import GREATER_THAN_THRESHOLD
```

Now define your `@acl.as_stop_criterion` with additional args `operator`:
```python
# Defining the stop criterion with a metric
@acl.as_stop_criterion(metric_name='custom_metric',
                       operator=GREATER_THAN_THRESHOLD, threshold=0.8)
async def check_metric(*args):
    return f'python3 check_custom_metric.py'
```

In this way, ROSE will understand the relation between the custom metric and the target threshold value.