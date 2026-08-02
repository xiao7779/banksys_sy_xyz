# banksys_sy_zhangxiaoying

银行营销数据交互式分析与在线认购预测系统。

## 功能

- **数据分析**:多维度可视化探索银行营销数据
- **在线预测**:基于机器学习模型的客户认购意向预测

## 技术栈

Python 3.11 · Streamlit · scikit-learn · plotly · pytest · ruff · Docker

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
python -m src.model_train

# 启动应用
streamlit run app.py --server.port 8888
```

## Docker 部署

```bash
docker build -t banksys-sy .
docker run -p 8888:8888 banksys-sy
```

访问 `http://localhost:8888`
