from flask import Flask, request, jsonify, send_from_directory
import ee
import os
import yfinance as yf
from flask_cors import CORS
import tempfile
from datetime import datetime, timedelta

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
        ticker = yf.Ticker("ZC=F")
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return []

        results = []
        for date, row in df.iterrows():
            results.append({
                "date": date.strftime('%Y-%m-%d'),
                "close": round(row['Close'], 2)
            })
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)