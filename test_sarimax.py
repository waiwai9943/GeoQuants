import sys
try:
    import statsmodels
    print(f"Statsmodels version: {statsmodels.__version__}")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas as pd
    import numpy as np
    
    # Try a simple fit
    data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32] * 5 # 60 points
    series = pd.Series(data)
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
    fit = model.fit(disp=False)
    print("SARIMAX fit successful")
    print(fit.summary().tables[0])
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other Error: {e}")
