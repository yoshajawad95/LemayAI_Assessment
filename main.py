from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import logging
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import os
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model_cache = {}
model_lock = asyncio.Lock()
class InferenceRequest(BaseModel):
    model_name: str
    inputs: str
    task: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}
class InferenceResponse(BaseModel):
    model_name: str
    outputs: Any
    processing_time: float
class ModelInfo(BaseModel):
    loaded_models: List[str]
    cache_size: int

async def load_model(model_name: str, task: Optional[str] = None):
    async with model_lock:
        if model_name not in model_cache:
            try:
                logger.info(f"Loading model: {model_name}")
                if not task:
                    if "sst-2" in model_name.lower() or "sentiment" in model_name.lower():
                        task = "sentiment-analysis"
                    elif "finetuned-sst" in model_name.lower():
                        task = "sentiment-analysis"
                    elif "gpt" in model_name.lower():
                        task = "text-generation"
                    elif "t5" in model_name.lower():
                        task = "text2text-generation"
                    elif "bert" in model_name.lower() or "roberta" in model_name.lower():
                        if "distilbert-base-uncased-finetuned-sst-2" in model_name.lower():
                            task = "sentiment-analysis"
                        else:
                            task = "fill-mask"
                    else:
                        task = "text-generation"
                
                pipe = pipeline(
                    task=task,
                    model=model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                model_cache[model_name] = {
                    "pipeline": pipe,
                    "task": task
                }
                logger.info(f"Successfully loaded model: {model_name} for task: {task}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HuggingFace Inference Server")
    default_model = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        await load_model(default_model, "sentiment-analysis")
        logger.info(f"Pre-loaded default model: {default_model}")
    except Exception as e:
        logger.warning(f"Could not pre-load default model: {e}")
    
    yield
    logger.info("Shutting down inference server")
    model_cache.clear()

app = FastAPI(
    title="HuggingFace Inference Server",
    description="A containerized server for running inference on Hugging Face model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "HuggingFace Inference Server",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "loaded_models": len(model_cache),
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/models", response_model=ModelInfo)
async def get_loaded_models():
    return ModelInfo(
        loaded_models=list(model_cache.keys()),
        cache_size=len(model_cache)
    )

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    import time
    start_time = time.time()
    
    try:
        if request.model_name not in model_cache:
            task_to_use = request.task if request.task else None
            await load_model(request.model_name, task_to_use)
        
        model_info = model_cache[request.model_name]
        pipe = model_info["pipeline"]
        
        logger.info(f"Running inference on {request.model_name}")
        
        if request.parameters:
            outputs = pipe(request.inputs, **request.parameters)
        else:
            outputs = pipe(request.inputs)
        
        processing_time = time.time() - start_time
        
        return InferenceResponse(
            model_name=request.model_name,
            outputs=outputs,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    async with model_lock:
        if model_name in model_cache:
            del model_cache[model_name]
            return {"message": f"Model {model_name} unloaded successfully"}
        else:
            raise HTTPException(status_code=404, detail="Model not found in cache")

@app.delete("/models")
async def clear_cache():
    """Clear all models from cache"""
    async with model_lock:
        model_cache.clear()
        return {"message": "All models cleared from cache"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)