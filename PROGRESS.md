# PROGRESS · 项目进度 〔本项目活记忆 · AI 维护〕

> **作用**:记录当前状态、下一步 TODO、关键决策(ADR)、踩坑记录(GOTCHAS)。
> **更新时机**:每完成一个模块或发生关键决策/故障时更新。

---

## 当前状态

- **所处阶段**:六步流程 第⑤步 — 本地 CI 自检通过，准备推送分支并发起 PR
- **最后更新**:2026-08-02

---

## 已完成

| # | 任务 | 状态 |
|---|---|---|
| 1 | 建仓:GitHub 仓库 `banksys_sy_zhangxiaoying` | ✅ |
| 2 | 工程骨架:`requirements.txt`、`requirements-dev.txt`、`src/`、`tests/` | ✅ |
| 3 | CI 流水线:`.github/workflows/ci.yml`(ruff+pytest+docker build) | ✅ |
| 4 | `src/data_loader.py` — 数据加载模块 + 12 个测试 | ✅ |
| 5 | `src/analysis.py` — 数据分析逻辑 + 11 个测试 | ✅ |
| 6 | `app.py` + `src/ui/page_analysis.py` — Streamlit 入口 + 数据分析页 | ✅ |
| 7 | `src/model_train.py` — 模型训练(Pipeline + 阈值优化) + 11 个测试 | ✅ |
| 8 | `src/predict.py` — 在线预测(使用最优阈值) + 5 个测试 | ✅ |
| 9 | `src/ui/page_prediction.py` — 在线预测页面(点选表单) + Dockerfile | ✅ |
| 10 | 本地 CI 自检:ruff format ✅ ruff check ✅ pytest 39/39 ✅ 覆盖率 99% ✅ | ✅ |

**模型业务指标**:
- AUC: 0.8963 ✅ (门禁≥0.75)
- 最优 F1: 0.6060 ✅ (门禁≥0.6, 使用最优阈值)
- 分类器: RandomForest(n_estimators=300, class_weight={0:1, 1:3})

---

## ADR(架构决策记录)

### ADR-1: 使用最优决策阈值替代默认 0.5
- **决策**:训练阶段通过 `precision_recall_curve` 寻找最大化 F1 的最优阈值，保存至模型文件，预测时使用该阈值。
- **原因**:数据不平衡(yes≈13%)，默认阈值 0.5 导致 F1 仅 0.29；优化后 F1 提升至 0.61。
- **日期**:2026-08-02

### ADR-2: 模型保存为 dict 结构
- **决策**:`save_model` 保存 `{"pipeline": Pipeline, "threshold": float}` 而非直接保存 Pipeline。
- **原因**:预测模块需要知道最优阈值，将其与模型管道打包存储。
- **日期**:2026-08-02

---

## GOTCHAS(踩坑记录)

_暂无。_
