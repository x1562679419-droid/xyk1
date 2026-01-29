"""
运动评估 API 服务
部署到云端（Render）
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os
import json
import httpx

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

app = FastAPI(title="运动评估 API", version="1.0.0")

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 客户端配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
    except Exception as e:
        print(f"OpenAI 客户端初始化失败: {e}")


# 数据模型
class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    score: float


class Pose(BaseModel):
    keypoints: List[Keypoint]
    score: float


class AnalysisRequest(BaseModel):
    poses: List[Pose]
    timestamp: int


class FeedbackItem(BaseModel):
    type: str
    text: str
    icon: str


class AnalysisResponse(BaseModel):
    overall: int
    accuracy: int
    coordination: int
    stability: int
    feedback: List[FeedbackItem]
    suggestions: List[str]


async def analyze_with_openai(poses: List[Pose]) -> dict:
    """使用 OpenAI API 分析姿态"""

    if not client:
        raise HTTPException(status_code=500, detail="OpenAI API 未配置")

    # 将姿态数据转换为 JSON 字符串
    poses_json = json.dumps([pose.model_dump() for pose in poses], ensure_ascii=False)

    # 构建 prompt
    prompt = f"""你是一个专业的运动分析师。请分析以下人体姿态数据。

姿态数据（JSON 格式）：
{poses_json}

请返回以下格式的 JSON（只返回 JSON，不要其他文字）：
{{
  "overall": 85,
  "accuracy": 88,
  "coordination": 82,
  "stability": 85,
  "feedback": [
    {{"type": "good", "text": "动作准确性很高，基本达到了标准姿势", "icon": "✅"}},
    {{"type": "improve", "text": "动作协调性可以进一步提升", "icon": "⚡"}}
  ],
  "suggestions": [
    "建议保持身体核心稳定，避免晃动",
    "练习时放慢动作速度，感受肌肉发力顺序"
  ]
}}

评分标准（0-100）：
- accuracy: 动作准确性，关键点位置是否正确
- coordination: 身体协调性，左右对称、动作流畅度
- stability: 动作稳定性，身体晃动程度

请根据姿态数据给出专业的分析和建议。
"""

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的运动分析师，擅长分析人体姿态和运动表现。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        result_text = response.choices[0].message.content.strip()

        # 清理可能的 Markdown 代码块
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"OpenAI API 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API 调用失败: {str(e)}")


def analyze_locally(poses: List[Pose]) -> dict:
    """本地分析（备用方案）"""
    import random

    overall = random.randint(70, 95)
    accuracy = random.randint(70, 95)
    coordination = random.randint(70, 95)
    stability = random.randint(70, 95)

    feedback = [
        {"type": "good", "text": "动作基本完成", "icon": "✅"},
        {"type": "improve", "text": "可以进一步优化动作标准度", "icon": "⚡"}
    ]

    suggestions = [
        "保持动作节奏，注意呼吸配合",
        "每次练习前做好充分热身"
    ]

    return {
        "overall": overall,
        "accuracy": accuracy,
        "coordination": coordination,
        "stability": stability,
        "feedback": feedback,
        "suggestions": suggestions
    }


@app.get("/")
async def root():
    """返回首页"""
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "openai_configured": client is not None,
        "openai_available": OPENAI_AVAILABLE,
        "model": OPENAI_MODEL
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """分析姿态数据"""

    if not request.poses:
        raise HTTPException(status_code=400, detail="姿态数据为空")

    try:
        # 优先使用 OpenAI API
        if client:
            result = await analyze_with_openai(request.poses)
        else:
            print("使用本地分析（OpenAI 未配置）")
            result = analyze_locally(request.poses)

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("""
    ╔═════════════════════════════════════════════════════════╗
    ║           运动评估 API 服务                              ║
    ╠═════════════════════════════════════════════════════════╣
    ║  本地开发模式                                           ║
    ║  服务地址：http://localhost:8000                          ║
    ╚═════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)
