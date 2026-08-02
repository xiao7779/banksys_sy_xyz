# PROGRESS · 项目进度 〔本项目活记忆 · AI 维护〕

> **作用**:记录当前状态、下一步 TODO、关键决策(ADR)、踩坑记录(GOTCHAS)。
> **更新时机**:每完成一个模块或发生关键决策/故障时更新。

---

## 当前状态

- **所处阶段**:六步流程 第①步 — 项目初始化(已填写 00/01/PROGRESS,待确认后建仓)
- **最后更新**:2026-08-02

---

## 第一批 TODO

| # | 任务 | 对应需求 | 优先级 |
|---|---|---|---|
| 1 | 建仓:GitHub 仓库 `banksys_sy_zhangxiaoying`,配置 `.gitignore`、`README.md` | US-1 | P0 |
| 2 | 搭建 Python 工程骨架:`requirements.txt`、`requirements-dev.txt`、`src/`、`tests/` | US-1 | P0 |
| 3 | 实现 CI 流水线:`.github/workflows/ci.yml`(ruff + pytest + docker build) | US-1 | P0 |
| 4 | 实现 `src/data_loader.py` — 数据加载模块,带单元测试 | US-2, US-3 | P0 |
| 5 | 实现 `src/analysis.py` — 数据分析逻辑(统计摘要、分组聚合),带单元测试 | US-2 | P0 |
| 6 | 实现 Streamlit 入口 `app.py` + 数据分析页面 `src/ui/page_analysis.py` | US-2 | P0 |
| 7 | 实现 `src/model_train.py` — 模型训练(预处理管道 + 分类器 + 评估),带单元测试 | US-3 | P0 |
| 8 | 实现 `src/predict.py` — 预测逻辑(模型加载 + 推理),带单元测试 | US-4 | P0 |
| 9 | 实现在线预测页面 `src/ui/page_prediction.py` | US-4 | P0 |
| 10 | 编写 Dockerfile,本地 `docker build && docker run` 验证 | US-1 | P0 |
| 11 | 本地 CI 自检(ruff + pytest + 覆盖率≥80%),全绿后提交 PR | US-1 | P1 |
| 12 | 模型训练并生成 `models/model.pkl`,端到端验证预测流程 | US-3, US-4 | P1 |

---

## ADR(架构决策记录)

_暂无。首个决策将在开发过程中记录。_

---

## GOTCHAS(踩坑记录)

_暂无。首次踩坑后将记录在此。_
