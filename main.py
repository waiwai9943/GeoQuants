from flask import Flask, request, jsonify, send_from_directory
import ee
import os
import json
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
        df = tickers.download(start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return []

        # Handle MultiIndex columns (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Drop ticker level if present

        df = df.reset_index()
        
        # Ensure we find the Date column
        date_col = None
        for col in df.columns:
            if str(col).lower() == 'date':
                date_col = col
                break
        
        if not date_col:
             if pd.api.types.is_datetime64_any_dtype(df.index):
                 df['date'] = df.index.strftime('%Y-%m-%d')
                 date_col = 'date'
             else:
                 for col in df.columns:
                     if pd.api.types.is_datetime64_any_dtype(df[col]):
                         df['date'] = df[col].dt.strftime('%Y-%m-%d')
                         date_col = 'date'
                         break
        else:
            df['date'] = df[date_col].dt.strftime('%Y-%m-%d')

        if not date_col or 'date' not in df.columns:
            print("Could not identify Date column")
            return []

        # Convert to list of dicts manually to ensure type safety
        results = []
        for _, row in df.iterrows():
            try:
                # Basic fields
                item = {
                    'date': str(row['date'])
                }
                
                # OHLC fields
                has_data = False
                for col_name in ['Open', 'High', 'Low', 'Close']:
                    lower_col = col_name.lower()
                    # Check original column name
                    if col_name in df.columns:
                        val = row[col_name]
                        # Handle potential Series/Ambiguity if multiple columns have same name
                        if isinstance(val, pd.Series):
                            val = val.iloc[0]
                        
                        if pd.notna(val):
                            item[lower_col] = round(float(val), 2)
                            has_data = True
                
                if has_data and 'close' in item:
                    results.append(item)
            except Exception as row_err:
                print(f"Skipping row due to error: {row_err}")
                continue
                
        return results

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

        # MASK NON-CROPLAND PIXELS (ESA WorldCover)
        # Class 40 = Cropland. This ensures we don't average forests/roads.
        world_cover = ee.ImageCollection("ESA/WorldCover/v100").first()
        cropland_mask = world_cover.select('Map').eq(40)
        
        # Apply mask to collection
        collection = collection.map(lambda img: img.updateMask(cropland_mask))

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

        # 3. STATISTICS (Correlation)
        stats = {}
        if sat_values and market_data:
            # Create DataFrames
            df_sat = pd.DataFrame({'date': pd.to_datetime(sat_dates), 'sat_val': sat_values})
            df_mkt = pd.DataFrame(market_data)
            df_mkt['date'] = pd.to_datetime(df_mkt['date'])
            
            # Merge on nearest date (within 5 days tolerance)
            df_merged = pd.merge_asof(df_sat.sort_values('date'), 
                                      df_mkt.sort_values('date'), 
                                      on='date', 
                                      direction='nearest', 
                                      tolerance=pd.Timedelta(days=5))
            
            # Drop NaN rows (unmatched dates)
            df_merged = df_merged.dropna()
            
            if len(df_merged) > 2:
                correlation = df_merged['sat_val'].corr(df_merged['close'])
                stats['pearson_correlation'] = round(correlation, 4)
                
                # Market Volatility (Std Dev of Close prices)
                volatility = df_mkt['close'].std()
                stats['market_volatility'] = round(volatility, 4)
                
                # Satellite Variability
                sat_std = df_sat['sat_val'].std()
                stats['sat_index_std'] = round(sat_std, 4)
            else:
                stats = {"message": "Not enough overlapping data for correlation."}

        return {
            "satelliteData": {
                "dates": sat_dates,
                "values": sat_values,
                "index": index_type
            },
            "marketData": market_data,
            "statistics": stats
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



SESSION_FILE = 'user_session.json'







@app.route('/save_session', methods=['POST'])



def save_session():



    try:



        data = request.json



        with open(SESSION_FILE, 'w') as f:



            json.dump(data, f)



        return jsonify({"status": "Session saved"})



    except Exception as e:



        return jsonify({"error": str(e)}), 500







@app.route('/load_session', methods=['GET'])



def load_session():



    if os.path.exists(SESSION_FILE):



        try:



            with open(SESSION_FILE, 'r') as f:



                data = json.load(f)



            return jsonify(data)



        except Exception as e:



            return jsonify({"error": str(e)}), 500



    return jsonify({})







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
