---
title: Interactive Data Visualization - Automobile Accidents
date: 2020-07-16T16:14:02+08:00
tags: [Python, Plotly]
share: true
comments: true
---

# Interactive Data Visualization

Automotive crashes continue to be one of the main reasons for American deaths. After seeing a decline in traffic fatalities for many years, 2015 saw an uptick in accidents. Many factors contributed to a higher number of accidents.

## Import Packages


```python
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Plotly is an online analytics and data visualization tool
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
import plotly
plotly.offline.init_notebook_mode()

```
