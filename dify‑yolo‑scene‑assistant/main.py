from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import time
import json
import requests

app = FastAPI(title="YOLO Harness Service")

# 模型路径，确认和best.pt位置匹配
MODEL_PATH = "./best.pt"
model = YOLO(MODEL_PATH)

LOG_FILE = "run_logs.jsonl"

def write_log(data: dict):
    """写入调用日志，harness评测用"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

@app.get("/health")
async def health():
    """健康检查接口，Dify可用来探测服务状态"""
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    start_time = time.time()
    log_entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "latency": None,
        "file_name": file.filename,
        "result": None
    }
    try:
        img_bytes = await file.read()
        results = model(img_bytes)
        res = results[0]

        boxes = []
        for box in res.boxes:
            boxes.append({
                "xyxy": box.xyxy.tolist()[0],
                "conf": float(box.conf),
                "cls": int(box.cls),
                "label": res.names[int(box.cls)]
            })

        output = {
            "success": True,
            "count": len(boxes),
            "boxes": boxes
        }
        log_entry["success"] = True
        log_entry["result"] = output

    except Exception as e:
        output = {
            "success": False,
            "error": str(e)
        }
        log_entry["result"] = output

    log_entry["latency"] = round(time.time() - start_time, 4)
    write_log(log_entry)
    return JSONResponse(content=output)


# ========== 新增：通过图片URL推理（适配Dify） ==========
@app.post("/detect_by_url")
async def detect_by_url(img_url: str):
    start_time = time.time()
    log_entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "latency": None,
        "url": img_url,
        "result": None
    }
    try:
        resp = requests.get(img_url, timeout=10)
        resp.raise_for_status()
        img_bytes = resp.content
        results = model(img_bytes)
        res = results[0]

        boxes = []
        for box in res.boxes:
            boxes.append({
                "xyxy": box.xyxy.tolist()[0],
                "conf": float(box.conf),
                "cls": int(box.cls),
                "label": res.names[int(box.cls)]
            })
        output = {
            "success": True,
            "count": len(boxes),
            "boxes": boxes
        }
        log_entry["success"] = True
        log_entry["result"] = output
    except Exception as e:
        output = {
            "success": False,
            "error": str(e)
        }
        log_entry["result"] = output

    log_entry["latency"] = round(time.time() - start_time,4)
    write_log(log_entry)
    return JSONResponse(content=output)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001)