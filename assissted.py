import asyncio
import logging
import joblib
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

executor = ThreadPoolExecutor()

with open("mapping_features.json", "r") as f:
    feature_mappings = json.load(f)
    
marital_status_mapping = feature_mappings['marital_status']
caste_mapping = feature_mappings['caste']
qualified_mapping = feature_mappings['highest_qualification']
employed_mapping = feature_mappings['employed']
occupation_mapping = feature_mappings['occupation']
on_behalf_mapping = feature_mappings['on_behalf']
state_mapping = feature_mappings['state']
city_mapping = feature_mappings['city']
income_mapping = feature_mappings['income']
height_mapping = feature_mappings['height']

xgb_model = joblib.load("xgb_membership_model.pkl")

with open("device_prices.json", "r") as f:
    device_prices_map = json.load(f)
    
device_prices_map = {str(k).lower().strip(): v for k, v in device_prices_map.items()}

city_list = [3, 6, 4, 2, 1, 0, 5]
state_list = [5, 0, 3, 1 , 8, 6, 2, 4, 7]
            
class PredictionRequest(BaseModel):
    member_id: int
    age: float
    gender: bool
    marital_status: int
    sect: bool
    caste: int
    income: int
    height: int
    highest_qualification: int
    employed: int
    occupation: int
    on_behalf: int
    ads: int
    device: Optional[str] = None
    present_country: int
    permanent_country: int
    present_state: int
    permanent_state:int
    present_city:int
    permanent_city: int
    family_info: bool
    preferences: bool

@app.post("/assissted_prediction")
async def assissted_prediction(data: PredictionRequest):
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, get_prediction, data)
        return result
    except Exception as e:
        logger.error(f"Error getting prediction results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction results: {str(e)}")

@app.post("/prediction_only")
async def prediction_only(data: PredictionRequest):
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, get_raw_prediction, data)
        return result
    except Exception as e:
        logger.error(f"Error getting prediction only: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction only: {str(e)}")

def get_prediction(data: PredictionRequest):

    device_key = (data.device or "").lower().strip()
    price = device_prices_map.get(device_key, 19990.0)
    if device_key not in device_prices_map:
        logger.info(f"Device '{device_key or 'missing'}' not found → default price 19990")
    device_val = 0 if price < 20000 else 1 if price < 40000 else 2 if price < 70000 else 3 if price < 100000 else 4

    if data.present_country != 101 or data.permanent_country != 101:
        state = 7
        city = 5
    else:
        state = max([state_mapping.get(str(data.present_state), 5), state_mapping.get(str(data.permanent_state), 5)], key=state_list.index)
        city = max([city_mapping.get(str(data.present_city), 3), city_mapping.get(str(data.permanent_city), 3)], key=city_list.index)

    features = (
        (data.age - 18) / 66,
        int(data.gender),
        marital_status_mapping.get(str(data.marital_status), 1),
        int(data.sect),
        caste_mapping.get(str(data.caste), 1),
        income_mapping.get(str(data.income), 0),
        height_mapping.get(str(data.height), 0.5),
        qualified_mapping.get(str(data.highest_qualification), 2),
        employed_mapping.get(str(data.employed), 2),
        occupation_mapping.get(str(data.occupation), 0),
        on_behalf_mapping.get(str(data.on_behalf), 3) ,
        data.ads,
        device_val,
        state,
        city,
        int(data.family_info),
        int(data.preferences)
    )
    
    prediction = xgb_model.predict_proba([features])[0][1]
    logger.info(f"Member ID: {data.member_id}, Prediction: {prediction}, Features: {features}")
    return 0 if prediction <= 0.1 else 1 if prediction < 0.5 else round(110 * (1 - math.exp(-2.25 * prediction)), 2)

def get_raw_prediction(data: PredictionRequest):
    device_key = (data.device or "").lower().strip()
    price = device_prices_map.get(device_key, 19990.0)
    if device_key not in device_prices_map:
        logger.info(f"Device '{device_key or 'missing'}' not found → default price 19990")
    device_val = 0 if price < 20000 else 1 if price < 40000 else 2 if price < 70000 else 3 if price < 100000 else 4

    if data.present_country != 101 or data.permanent_country != 101:
        state = 7
        city = 5
    else:
        state = max([state_mapping.get(str(data.present_state), 5), state_mapping.get(str(data.permanent_state), 5)], key=state_list.index)
        city = max([city_mapping.get(str(data.present_city), 3), city_mapping.get(str(data.permanent_city), 3)], key=city_list.index)

    features = (
        (data.age - 18) / 66,
        int(data.gender),
        marital_status_mapping.get(str(data.marital_status), 1),
        int(data.sect),
        caste_mapping.get(str(data.caste), 1),
        income_mapping.get(str(data.income), 0),
        height_mapping.get(str(data.height), 0.5),
        qualified_mapping.get(str(data.highest_qualification), 2),
        employed_mapping.get(str(data.employed), 2),
        occupation_mapping.get(str(data.occupation), 0),
        on_behalf_mapping.get(str(data.on_behalf), 3) ,
        data.ads,
        device_val,
        state,
        city,
        int(data.family_info),
        int(data.preferences)
    )
    
    prediction = xgb_model.predict_proba([features])[0][1]
    logger.info(f"Member ID: {data.member_id}, Raw Prediction: {prediction}, Raw Features: {features}")
    return float(prediction) * 100

#uvicorn assissted:app --host 0.0.0.0 --port 2000 --workers 1

#Server Side
#source venv/bin/activate
#ps aux | grep assissted | grep -v grep
#kill -9 3205276
# nohup uvicorn assissted:app --host 0.0.0.0 --port 2000 --workers 1 \
# > assisted_uvicorn.log 2>&1 &



