from os import stat
from fastapi import APIRouter, Depends, status, Response
from utils import NER_label_model
from typing import Optional

from pydantic import BaseModel

from typing import Any, Dict, AnyStr, List, Union
from db.mongodb import get_database
from core.config import DATABASE_NAME, LABEL_COLLECTION

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."



class predict_text_label_body(BaseModel):
    return_top_n: Optional[int] = -1
    return_higher_than: Optional[float] = 0
    wanted_labels: Optional[list] = ["*"]
    specify_model_version: Optional[dict] = {}
    text: str = example_text


NER_LABEL_PREDICT_API_TAG = ["Predict"]
@router.post("/models/NER/labelText", tags = NER_LABEL_PREDICT_API_TAG, status_code=status.HTTP_200_OK)
def predict_text_label(response: Response, data: predict_text_label_body):
    NER_label_model.check_and_update_adapter()
    if data.specify_model_version not in [None, {}]:
        # Check model key is OK
        want_adapter = set(data.specify_model_version.keys())
        have_adapter = set(NER_label_model.global_all_adapter_names)
        not_include = want_adapter - have_adapter
        if not_include:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {
                "message": f"wanted_labels {not_include} not found."
            }
        
        check_pass, not_found = NER_label_model.check_adapters_version_available(
            data.specify_model_version)
        if check_pass == False:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {
                "not_found": not_found,
            }
        
        want_adapters = data.specify_model_version.values()
        NER_label_model.update_global_adapters_filenames(data.specify_model_version)
        NER_label_model.load_adapters(want_adapters)
        NER_label_model.update_model_active_head()
    
    if data.return_top_n <= 0:
        data.return_top_n = None

    unsupport = set(data.wanted_labels) - set(NER_label_model.global_all_adapter_names)
    if data.wanted_labels == ["*"]:
        data.wanted_labels = NER_label_model.model.active_head
    elif unsupport:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "message": f"wanted_labels {unsupport} not found."
        }

    sen, pred, logits, logits_order = NER_label_model.predict(data.text)
    out_Tokens = []
    for i, _ in enumerate(sen):
        predictions = []
        for j, _ in enumerate(logits_order):
            if logits_order[j] in data.wanted_labels:
                predictions.append({
                    "type": logits_order[j],
                    "confidence": logits[i][j],
                })
        predictions.sort(key = lambda x: x["confidence"], reverse=True)
        predictions = list(filter(lambda x: x["confidence"] > data.return_higher_than, predictions))
        out_Tokens.append({
            "token": sen[i],
            "predictions": predictions[:data.return_top_n],
        })
    return {
        "prediction": out_Tokens
    }

@router.get("/models/NER/labelText", tags = NER_LABEL_PREDICT_API_TAG, status_code=status.HTTP_200_OK)
def get_NER_labelText_model_status(response: Response):
    
    NER_label_model.check_and_update_adapter()
    return {
        "message": "get success",
        "status": "online",
        "label": NER_label_model.model.active_head,
        "version": NER_label_model.global_adapters_filenames,
    }

class specify_NER_labelText_model_version_body(BaseModel):
    label_name: str = "O"
    model_version: str = "newest"

@router.put("/models/NER/labelText", tags = NER_LABEL_PREDICT_API_TAG)
async def specify_NER_labelText_model_version(response: Response, data: specify_NER_labelText_model_version_body):
    mongo_client = await get_database()
    col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]
    result = await col.find_one({
        "label_name": data.label_name
    })
    if result == None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "message": "model not found",
            "not_found": data.label_name,
        }
    if data.model_version == "newest":
        history: list = result["adapter"]["history"]
        history.sort(key = lambda x: x["time"], reverse=True)
        data.model_version = history[0]["filename"]
    else:
        check_pass = NER_label_model.have_adapter_version(
            data.label_name, data.model_version)
        if check_pass == False:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {
                "not_found": {data.label_name:data.model_version},
            }
    
    result = await col.update_one(
        {"label_name": data.label_name},
        {"$set": {
            "adapter.lastest_filename": data.model_version
        }})
    if result.modified_count:
        response.status_code = status.HTTP_200_OK
        return {
            "message": "OK",
        }
    else:
        response.status_code = status.HTTP_304_NOT_MODIFIED
        return {
            "message": "modified_count = 0"
        }