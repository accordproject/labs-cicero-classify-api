def convert_mongo_id(data):
    data["_id"] = str(data["_id"])
    return data