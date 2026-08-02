# 00 · 项目上下文 〔本项目活记忆 · AI 维护〕

> **作用**:这是项目的"身份档案"。AI 接管项目时先读这里,了解项目目标、技术栈、目录、部署取值。
> **更新时机**:架构、技术栈、目录结构、端口、部署目录、重要约束变化时更新。

---

## 1. 项目是什么

- **项目名称**:`banksys_sy_zhangxiaoying`
- **一句话目标**:基于银行营销数据,提供交互式数据分析与在线预测(认购意向)的 Web 应用。
- **使用者/受益者**:银行业务人员 / 数据分析师,用于探索营销数据特征并预测客户是否会认购定期存款产品。
- **核心功能**:
  - **数据分析交互页面**:对银行营销数据进行多维度可视化探索(分布、相关性、趋势等)。
  - **在线预测系统**:基于历史数据离线训练分类模型,用户通过点选表单输入客户特征,实时返回"是否会认购"的预测结果。
- **输入/数据**:
  - 来源:`data/train.csv`(训练集,22.5k 行)、`data/test.csv`(测试集,7.5k 行)
  - 特征字段:age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_index, cons_conf_index, lending_rate3m, nr_employed
  - 目标字段:`subscribe`(yes/no)
  - 敏感程度:公开银行营销数据集(非个人真实数据)
  - 是否进 Git:数据不进 Git(通过 `.gitignore` 排除);模型产物(`.pkl` 等)不进 Git

## 2. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 用户指定 |
| Web/可视化框架 | Streamlit | 用户指定;适合数据应用快速搭建,内置交互组件 |
| 数据处理 | pandas, numpy | 数据清洗、特征工程 |
| 机器学习 | scikit-learn | 分类模型训练与评估 |
| 可视化 | plotly / matplotlib | 交互式图表;plotly 与 Streamlit 集成好 |
| 测试 | pytest | 用户指定 |
| 格式/静态检查 | ruff (format + check) | 用户指定 |
| 打包/运行 | Docker | 用户指定;本地部署 |
| CI | GitHub Actions | 课程统一要求;仅 CI,不做 CD |

## 3. 目录地图

```text
banksys_sy_zhangxiaoying/
├── standards/                     # AI 项目记忆与通用规范
│   ├── README.md
│   ├── 00-project-context.md
│   ├── 01-requirements.md
│   ├── 02-coding-standards.md
│   ├── 03-testing-standards.md
│   ├── 04-git-workflow.md
│   ├── 05-cicd-standards.md
│   ├── 06-ai-collab-protocol.md
│   └── templates/
├── data/                          # 原始数据(不进 Git)
│   ├── train.csv
│   └── test.csv
├── src/                           # 源码
│   ├── __init__.py
│   ├── data_loader.py             # 数据加载
│   ├── analysis.py                # 数据分析逻辑
│   ├── model_train.py             # 模型训练
│   ├── predict.py                 # 预测逻辑
│   └── ui/                        # Streamlit 页面
│       ├── __init__.py
│       ├── page_analysis.py       # 数据分析页
│       └── page_prediction.py     # 在线预测页
├── app.py                         # Streamlit 入口
├── models/                        # 训练产出的模型文件(不进 Git)
├── tests/                         # 测试
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   ├── test_model_train.py
│   └── test_predict.py
├── requirements.txt               # 生产运行依赖
├── requirements-dev.txt           # 本地/CI 检查依赖
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .github/workflows/
│   └── ci.yml
├── PROGRESS.md
└── README.md
```

> 新增目录前先更新本节,避免项目越做越散。

## 4. 质量门槛

| 类型 | 本项目标准 |
|---|---|
| 格式检查 | `ruff format --check .` |
| 静态检查 | `ruff check .` |
| 单元测试 | `pytest` |
| 覆盖率 | ≥ 80% |
| 构建 | `docker build` 成功(仅 CI,本地不强制) |
| 业务/模型指标 | 模型 AUC ≥ 0.75、F1 ≥ 0.6(分类:subscribe yes/no) |

## 5. 不变约束

- 密钥、密码、私钥、Token **绝不写进代码或文档**,只进 GitHub Secrets / 环境变量。
- 大文件、数据集(`data/*.csv`)、模型产物(`models/*.pkl`)不进 Git,通过 `.gitignore` 排除。
- `main` 分支受保护,日常开发必须走 feature 分支 + PR。
- CI 红灯不合并。
- 本项目只做 CI,不做 CD;本地 Docker 部署验证。

## 6. 部署/CI 占位符取值

| 占位符 | 本项目取值 | 说明 |
|---|---|---|
| `<APP>` | `banksys-sy` | 应用名/镜像名/容器名 |
| `<DEPLOY_DIR>` | 本地部署,无需远程目录 | 仅本地 `docker run` |
| `<PORT>` | `8888` | 服务端口(用户指定) |
| `<PYVER>` | `3.11` | Python 版本 |
| `<HEALTHCHECK>` | `/health` (Streamlit 默认 `/` 可达即可) | Streamlit 无独立 health 端点,以首页 HTTP 200 为健康标志 |
| `<SSH_USER>` | 不适用 | 无远程部署 |
| `<SSH_HOST>` | 不适用 | 无远程部署 |
