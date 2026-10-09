"""
FastAPI 服务
提供双任务情感分析 HTTP 接口：

二分类（正/负面）：
- POST /predict            单条
- POST /predict_batch      批量

情绪 6 分类（开心/感激/悲伤/愤怒/恐惧/焦虑）：
- POST /predict_emotion        单条（含 Top-3 情绪与极性兼容字段，自动入历史库）
- POST /predict_emotion_batch  批量

长文本：
- POST /analyze_long_text      分句情绪曲线 + 聚合（自动入历史库）

历史与纠错闭环：
- GET  /history                历史列表
- GET  /history/stats          历史/纠错计数
- POST /feedback               提交纠正标签
- GET  /feedback/export        导出纠错样本 CSV（列与训练集对齐）
"""
import os
import sys
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.deploy.predictor import get_predictor
from src.deploy.emotion_predictor import (
    get_emotion_predictor, predict_routed, predict_batch_routed,
)
from src.deploy.long_text import analyze_long_text
from src.storage import history as history_store

# /feedback 写操作鉴权：设置环境变量 APP_API_TOKEN 后启用，
# 请求需携带 Header「X-API-Token」。未设置时视为本地开发模式（放行）。
API_TOKEN = os.environ.get("APP_API_TOKEN", "")

# 输入上限（防止超长文本/超大批量拖垮服务）
MAX_TEXT_LEN = 2000
MAX_BATCH_ITEMS = 200

app = FastAPI(
    title="中文情感分析 API",
    description="基于 BERT 的中文评论分析：正/负二分类 + 6 类情绪 + 长文本曲线 + 反馈闭环",
    version="3.1.0",
)


# ===== 请求/响应数据模型 =====
TextField = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LEN,
                                 description=f"待分析的文本（≤{MAX_TEXT_LEN} 字）")]


class PredictRequest(BaseModel):
    text: TextField
    # 情绪6分类后端选择（二分类/长文本接口忽略此字段）：
    # bert=本地 BERT（默认，零回归），rag=BERT+DeepSeek 检索精判，lora=QLoRA 生成式
    model_type: Literal["bert", "rag", "lora"] = "bert"


class BatchPredictRequest(BaseModel):
    texts: List[TextField] = Field(
        ..., min_length=1, max_length=MAX_BATCH_ITEMS,
        description=f"待分析的文本列表（≤{MAX_BATCH_ITEMS} 条）",
    )
    model_type: Literal["bert", "rag", "lora"] = "bert"


class FeedbackRequest(BaseModel):
    record_id: int = Field(..., description="历史记录 id")
    correct_label: str = Field(..., description="纠正后的情绪英文键，如 angry")


def _check_feedback_token(x_api_token: Optional[str]) -> None:
    """feedback 写操作鉴权：未配置 APP_API_TOKEN 时放行（本地开发模式）"""
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Token")


# ===== 启动时加载模型（任一模型缺失不阻塞另一个）=====
@app.on_event("startup")
async def startup_event():
    history_store.init_db()
    try:
        get_predictor()
        print("[就绪] 二分类模型")
    except Exception as e:
        print(f"[跳过] 二分类模型加载失败: {e}")
    try:
        get_emotion_predictor()
        print("[就绪] 情绪6分类模型")
    except Exception as e:
        print(f"[跳过] 情绪模型加载失败（可先运行 train_bert_emotion.py）: {e}")
    print("服务就绪！")


# ===== 通用接口 =====
@app.get("/health", summary="健康检查", tags=["通用"])
async def health():
    return {"status": "ok", "service": "sentiment-analysis", "version": "3.1.0"}


# ===== 二分类接口 =====
@app.post("/predict", summary="单条正/负面预测", tags=["二分类"])
async def predict(request: PredictRequest):
    try:
        result = get_predictor().predict(request.text)
        result["record_id"] = history_store.add_record("binary", request.text, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict_batch", summary="批量正/负面预测", tags=["二分类"])
async def predict_batch(request: BatchPredictRequest):
    try:
        results = get_predictor().predict_batch(request.texts)
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


# ===== 情绪 6 分类接口 =====
@app.post("/predict_emotion", summary="单条情绪预测（6类 + Top3 + 自动存档，可选 model_type）", tags=["情绪6分类"])
async def predict_emotion(request: PredictRequest):
    try:
        result = predict_routed(request.text, request.model_type)
        result["record_id"] = history_store.add_record("emotion", request.text, result)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"情绪预测失败: {str(e)}")


@app.post("/predict_emotion_batch", summary="批量情绪预测（6类，可选 model_type）", tags=["情绪6分类"])
async def predict_emotion_batch(request: BatchPredictRequest):
    try:
        results = predict_batch_routed(request.texts, request.model_type)
        return {"results": results, "total": len(results),
                "model_type": request.model_type}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量情绪失败: {str(e)}")


# ===== 长文本接口 =====
@app.post("/analyze_long_text", summary="长文本分句情绪曲线（自动存档）", tags=["长文本"])
async def analyze_long_text_api(request: PredictRequest):
    try:
        result = analyze_long_text(request.text, get_emotion_predictor())
        if result["sentences"]:
            result["record_id"] = history_store.add_record("long_text", request.text, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"长文本分析失败: {str(e)}")


# ===== 历史与纠错闭环接口 =====
@app.get("/history", summary="查询分析历史", tags=["历史与纠错"])
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    task_type: Optional[str] = Query(None, regex="^(emotion|binary|long_text)$"),
    corrected_only: bool = False,
):
    return {"records": history_store.list_records(limit, task_type, corrected_only)}


@app.get("/history/stats", summary="历史/纠错计数", tags=["历史与纠错"])
async def get_history_stats():
    return history_store.stats()


@app.post("/feedback", summary="提交纠正标签（需 X-API-Token，如已配置）", tags=["历史与纠错"])
async def submit_feedback(request: FeedbackRequest, x_api_token: Optional[str] = Header(None)):
    _check_feedback_token(x_api_token)
    try:
        ok = history_store.update_feedback(request.record_id, request.correct_label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"记录 {request.record_id} 不存在")
    return {"status": "ok", "record_id": request.record_id,
            "corrected_label": request.correct_label}


@app.get("/feedback/export", summary="导出纠错样本 CSV（训练集格式）", tags=["历史与纠错"])
async def export_feedback():
    path, count = history_store.export_feedback_csv()
    if count == 0:
        raise HTTPException(status_code=404, detail="暂无可导出的纠错样本（纠正标签与原预测一致或未纠错）")
    return FileResponse(
        path, filename=path.name, media_type="text/csv; charset=utf-8"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
