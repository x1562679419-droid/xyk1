# 运动评估 Web 应用

AI 驱动的运动姿态分析工具，支持实时摄像头捕捉和图片上传分析。

## 功能特性

- 📷 摄像头实时捕捉
- 📤 图片上传分析
- 🤖 AI 智能分析（基于 MoveNet）
- 📊 多维度评分（准确性、协调性、稳定性）
- 💡 个性化改进建议

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
python main.py

# 访问 http://localhost:8000
```

## 部署到 Render

1. 将代码推送到 GitHub 仓库

2. 在 [Render](https://render.com) 创建新的 Web Service

3. 连接 GitHub 仓库

4. 配置环境变量：
   ```
   OPENAI_API_KEY=your_api_key
   OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
   OPENAI_MODEL=doubao-pro-4k
   ```

5. 启动命令：
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

## API 接口

### POST /api/analyze

分析姿态数据

**请求：**
```json
{
  "poses": [
    {
      "keypoints": [
        {"name": "nose", "x": 320, "y": 150, "score": 0.95}
      ],
      "score": 0.9
    }
  ],
  "timestamp": 1234567890
}
```

**响应：**
```json
{
  "overall": 85,
  "accuracy": 88,
  "coordination": 82,
  "stability": 85,
  "feedback": [
    {"type": "good", "text": "...", "icon": "✅"}
  ],
  "suggestions": ["..."]
}
```

## 技术栈

- 前端：HTML5 + TensorFlow.js + MoveNet
- 后端：FastAPI + Uvicorn
- AI：OpenAI API（火山引擎）

## License

MIT
