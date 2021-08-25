from fastapi import APIRouter, Depends, status, Response
from utils import NER_label_model
from typing import Optional

from pydantic import BaseModel

from typing import Any, Dict, AnyStr, List, Union

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

router = APIRouter()

example_text = "Dan Will be deemed to have completed its delivery obligations before 2021-7-5 if in Niall's opinion, the Jeep Car satisfies the Acceptance Criteria, and Niall notifies Dan in writing that it is accepting the Jeep Car."



class text_label_body(BaseModel):
    return_top_n: Optional[int] = -1
    return_higher_than: Optional[float] = 0
    wanted_labels: Optional[list] = ["*"]
    text: str = example_text


NER_LABEL_PREDICT_API_TAG = ["Predict"]
@router.post("/models/NER/labelText", tags = NER_LABEL_PREDICT_API_TAG, status_code=status.HTTP_200_OK)
def text_label(response: Response, data: text_label_body):
    NER_label_model.check_and_update_new_adapter()
    
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
def get_model_status(response: Response):
    
    NER_label_model.check_and_update_new_adapter()
    return {
        "message": "get success",
        "status": "online",
        "label": NER_label_model.model.active_head,
        "version": NER_label_model.global_adapters_filenames,
    }
