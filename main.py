from flask import Flask, request, jsonify, send_from_directory
import ee
import os
import yfinance as yf
from flask_cors import CORS
import tempfile
from datetime import datetime, timedelta
import pandas as pd
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None


# =========================================================================================
# INITIALIZATION
# =========================================================================================

credentials = None
SERVICE_AC = "user-374@agri-471404.iam.gserviceaccount.com" 
PROJECT_ID = "agri-471404" 

def initialize_gee_credentials():
    global credentials
    gee_key_json = os.environ.get('GEE_KEY_JSON')
    
    if gee_key_json:
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
                tmp_file.write(gee_key_json)
                key_path = tmp_file.name
            credentials = ee.ServiceAccountCredentials(SERVICE_AC, key_path)
            os.remove(key_path)
        except Exception as e:
            credentials = None
    else:
        # Local fallback
        credentials_path = 'credentials/agri-471404-201b5260c966.json'
        if os.path.exists(credentials_path):
             credentials = ee.ServiceAccountCredentials(SERVICE_AC, credentials_path)

    if credentials:
        try:
            ee.Initialize(credentials, project=PROJECT_ID)
            print("Google Earth Engine initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize GEE: {e}")
            credentials = None

initialize_gee_credentials()

app = Flask(__name__, static_folder='.')
CORS(app)

# =========================================================================================
# HELPER FUNCTIONS
# =========================================================================================

def add_indices(image):
    """Calculates both NDVI and NDRE (Red Edge)."""
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    # NDRE = (NIR - RedEdge1) / (NIR + RedEdge1) -> Critical for Corn
    ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
    return image.addBands([ndvi, ndre])

def get_corn_futures(start_date, end_date):
    """Fetches Corn Futures (ZC=F) data from Yahoo Finance."""
    try:
        # ZC=F is the ticker for Corn Futures on Yahoo Finance
        tickers = yf.Tickers("ZC=F")
        df = tickers.download(start=start_date, end=end_date)
        
        if df.empty:
            return []

        # Handle MultiIndex columns (common in new yfinance)
        # If columns are MultiIndex, accessing 'Close' might return a DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Drop ticker level if present

        df = df.reset_index()
        
        # Ensure we find the Date column (sometimes it's index, sometimes column)
        date_col = None
        for col in df.columns:
            if str(col).lower() == 'date':
                date_col = col
                break
        
        if not date_col:
             # Fallback if reset_index failed to name it 'Date'
             if pd.api.types.is_datetime64_any_dtype(df.index):
                 df['date'] = df.index.strftime('%Y-%m-%d')
             else:
                 # Try to find any datetime column
                 for col in df.columns:
                     if pd.api.types.is_datetime64_any_dtype(df[col]):
                         df['date'] = df[col].dt.strftime('%Y-%m-%d')
                         break
        else:
            df['date'] = df[date_col].dt.strftime('%Y-%m-%d')

        if 'date' not in df.columns:
            print("Could not identify Date column")
            return []

        # Handle Close column
        if 'Close' in df.columns:
            # Ensure it is a Series (handle potential DataFrame if multiple cols named Close existed)
            close_data = df['Close']
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]
            
            # Convert to native float to ensure JSON serializability
            df['close'] = close_data.apply(lambda x: round(float(x), 2))
        else:
             print("Close column not found")
             return []
        
        return df[['date', 'close']].to_dict('records')

    except Exception as e:
        print(f"Yahoo Finance Error: {e}")
        return []

# =========================================================================================
# ANALYSIS LOGIC
# =========================================================================================

def analyze_corn_health(polygon, start_date, end_date, index_type='NDRE'):
    try:
        # 1. SATELLITE DATA (GEE)
        aoi = ee.Geometry.Polygon(polygon['coordinates'])
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 5-day steps to catch clear images
        step_days = 5 
        time_step = timedelta(days=step_days)

        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .map(add_indices)

        sat_dates = []
        sat_values = []
        current = start_dt

        # Iterate through time windows
        while current < end_dt:
            next_step = current + time_step
            if next_step > end_dt: next_step = end_dt
            
            s_str = current.strftime('%Y-%m-%d')
            e_str = next_step.strftime('%Y-%m-%d')
            
            period_col = collection.filterDate(s_str, e_str)
            
            if period_col.size().getInfo() > 0:
                comp = period_col.median().clip(aoi)
                val = comp.select(index_type).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=20,
                    maxPixels=1e9
                ).get(index_type).getInfo()
                
                if val is not None:
                    sat_dates.append(s_str)
                    sat_values.append(round(val, 4))
            
            current = next_step

        # 2. FINANCIAL DATA (Yahoo Finance)
        market_data = get_corn_futures(start_date, end_date)

        return {
            "satelliteData": {
                "dates": sat_dates,
                "values": sat_values,
                "index": index_type
            },
            "marketData": market_data
        }

    except Exception as e:
        print(f"Analysis failed: {e}")
        return {"error": str(e)}



# =========================================================================================

# PREDICTION LOGIC

# =========================================================================================



def predict_vegetation_sarima(dates, values):

    """

    Predicts future vegetation index values using a SARIMA model.

    

    SARIMA model parameters (p,d,q)(P,D,Q,s) are heuristics and should be tuned for better results,

    for example, by using an ACF/PACF plot analysis or a grid search (e.g., pmdarima library).

    

    - (p,d,q): Non-seasonal order for ARIMA.

    - (P,D,Q,s): Seasonal order for ARIMA.

    - s: The number of time steps for a single seasonal period. Data is in 5-day intervals,

         so a yearly seasonality is approx. 365/5 = 73. We use 12 as a simpler starting point

         assuming some monthly pattern, but this is a key parameter to tune.

    """

    if not SARIMAX:

        raise ImportError("statsmodels is not installed. Please install it with 'pip install statsmodels'")



    if len(values) < 24: # Need enough data for seasonal model

        return {"error": "Not enough data points to perform a forecast. At least 24 points are recommended."}



    # Create a pandas Series from the data

    # The data is already in 5-day intervals, which is a regular frequency.

    series = pd.Series(values, index=pd.to_datetime(dates))



    # SARIMA model - using some example parameters. These should be tuned.

    # Order (p,d,q)

    order = (1, 1, 1) 

    # Seasonal Order (P,D,Q,s)

    # We assume a yearly seasonality, with 's' being the number of periods in a season.

    # With 5-day intervals, s would be ~73 for a year. Let's use 12 as a simpler proxy for monthly patterns.

    seasonal_order = (1, 1, 1, 12) 

    

    try:

        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,

                        enforce_stationarity=False, enforce_invertibility=False)

        

        fit = model.fit(disp=False)

        

        # Forecast for the next 6 steps (i.e., next 30 days)

        forecast_steps = 6

        forecast = fit.get_forecast(steps=forecast_steps)

        

        # Get forecast values and confidence intervals

        pred_values = forecast.predicted_mean.tolist()

        conf_int = forecast.conf_int()



        # Generate forecast dates

        last_date = series.index[-1]

        forecast_dates = [(last_date + timedelta(days=5 * (i + 1))).strftime('%Y-%m-%d') for i in range(forecast_steps)]



        return {

            "dates": forecast_dates,

            "values": pred_values,

            "conf_int_lower": conf_int.iloc[:, 0].tolist(),

            "conf_int_upper": conf_int.iloc[:, 1].tolist()

        }

    except Exception as e:

        print(f"SARIMA prediction failed: {e}")

        # Providing a more user-friendly error

        if "LinAlgError" in str(e):

             return {"error": "Prediction failed due to numerical instability. The data may be too uniform or short."}

        return {"error": f"An error occurred during prediction: {e}"}





# =========================================================================================

# ROUTES

# =========================================================================================



@app.route('/')

def index():

    return send_from_directory('.', 'index.html')



@app.route('/analyze_corn', methods=['POST'])

def analyze_corn():

    if not credentials:

        return jsonify({"error": "GEE credentials missing."}), 500



    data = request.json

    polygon = data.get('polygon')

    start = data.get('startDate')

    end = data.get('endDate')

    idx = data.get('indexType', 'NDRE') 



    if not polygon:

        return jsonify({"error": "Missing polygon"}), 400



    result = analyze_corn_health(polygon, start, end, idx)

    

    if "error" in result:

        return jsonify(result), 400

        

    return jsonify(result)



@app.route('/predict_vegetation', methods=['POST'])



def predict_vegetation():



    if not SARIMAX:



        return jsonify({"error": "Prediction library not installed on server."}), 500







    data = request.json



    dates = data.get('dates')



    values = data.get('values')







    if not dates or not values:



        return jsonify({"error": "Missing dates or values for prediction"}), 400







    result = predict_vegetation_sarima(dates, values)



    



    if "error" in result:



        return jsonify(result), 400



        



    return jsonify(result)











@app.route('/health')



def health_check():



    return jsonify({"status": "OK"})





if __name__ == '__main__':

    if not SARIMAX:

        print("WARNING: 'statsmodels' library not found. Prediction endpoint will not work.")

        print("Please install it with: pip install statsmodels")

    port = int(os.environ.get('PORT', 8080))

    app.run(host='0.0.0.0', port=port)
