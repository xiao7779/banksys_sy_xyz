# 01 · 需求 / 活 PRD 〔本项目活记忆 · AI 维护〕

> **作用**:这是本项目唯一的需求文档。所有新功能、缺陷、技术债都追加到这里,不要另起多个 PRD 文件。
> **更新时机**:每次有新需求、需求变更、验收标准变化时更新。

---

## 1. 需求来源

| 类型 | 来源 | 进入方式 |
|---|---|---|
| 功能需求 Feature | 用户 / 老师 / 产品 | 写成用户故事 |
| 缺陷 Bug | 测试 / 线上日志 / 用户反馈 | 写复现步骤和期望结果 |
| 技术债 Tech Debt | 开发 / Review / CI/CD 故障 | 写影响和修复目标 |

---

## 2. Issue 生命周期

| 阶段 | 状态 | 动作 |
|---|---|---|
| 提出 | Open | 写清场景、目标、验收标准 |
| 排期 | Backlog / Todo | 决定优先级和负责人 |
| 开发 | In Progress | 从 main 开 feature 分支 |
| 评审 | In Review | 提 PR,等待 CI 和 Review |
| 合并 | Done | PR 合并 main,自动关闭 Issue |
| 验收 | Verified | 按验收标准确认 |

**追踪规则**:分支名带 Issue 号,PR 描述写 `closes #<编号>`。

---

## 3. 用户故事模板

```text
### US-<编号> <一句话标题> · 状态: Backlog
作为 <角色>,
我想要 <能力>,
以便 <价值>。

验收标准:
- AC1: Given <前提>,When <动作>,Then <可验证结果>。
- AC2: <补充标准>

技术备注:
- <可选:约束、边界、风险>
```

---

## 4. 需求清单

### US-1 初始化项目工程化与 CI · 状态: Backlog

作为 **项目开发者**,
我想要 项目具备基础工程结构、本地测试能力与 GitHub Actions CI,
以便 后续每次开发都能自动检查代码质量。

验收标准:
- AC1: Given 空仓库,When 从 main 开 feature 分支完成初始化,Then 目录结构符合 `00-project-context.md` 的目录地图。
- AC2: Given 提交 PR,When CI 触发,Then 包含 ruff format、ruff check、pytest --cov(覆盖率≥80%)、docker build。
- AC3: Given 本地开发完成后运行 `pytest`,Then 所有测试通过,覆盖率≥80%。
- AC4: Given 服务启动,When 访问 `http://localhost:8888`,Then 可以看到 Streamlit 首页。
- AC5: Given `docker build -t banksys-sy . && docker run -p 8888:8888 banksys-sy`,When 访问 `http://localhost:8888`,Then 服务正常响应(HTTP 200)。
- AC6: Given 项目根目录,When 运行 `ruff format --check . && ruff check .`,Then 零错误零警告。

技术备注:
- 本项目只做 CI,不做 CD(无远程服务器部署)。
- 数据文件 `data/*.csv` 和模型文件 `models/*.pkl` 不进 Git。
- CI 中的 docker build 在云端 runner 执行(本地不强制)。

---

### US-2 数据分析交互页面 · 状态: Backlog

作为 **银行业务分析人员**,
我想要 在 Web 页面上交互式地探索银行营销数据,
以便 快速了解客户特征分布、各因素与认购行为的关系,为业务决策提供依据。

验收标准:
- AC1: Given 应用已启动,When 访问数据分析页面,Then 页面展示数据总览(总行数、特征数、认购率等关键统计量)。
- AC2: Given 数据分析页面,When 用户选择某个数值特征(如 age、duration),Then 显示该特征的分布直方图与箱线图,并按 subscribe 分组着色。
- AC3: Given 数据分析页面,When 用户选择某个类别特征(如 job、marital、education),Then 显示该特征的频次柱状图及各分类的认购率。
- AC4: Given 数据分析页面,When 用户查看相关性分析,Then 展示数值特征之间的相关性热力图。
- AC5: Given 数据分析页面,When 页面加载完成,Then 所有图表可交互(缩放、悬停查看数值、图例切换)。
- AC6: Given 数据分析页面,When 多次切换/选择不同特征,Then 页面响应时间不超过 3 秒。

技术备注:
- 使用 plotly 绘制交互式图表,嵌入 Streamlit。
- 数据分析仅在页面展示,不做数据导出。
- 数据加载逻辑独立封装在 `src/data_loader.py`,供分析页和训练模块复用。

---

### US-3 模型离线训练 · 状态: Backlog

作为 **数据科学家**,
我想要 基于历史营销数据离线训练一个二分类模型(预测 subscribe),
以便 模型达到可接受的预测性能,为在线预测系统提供推理能力。

验收标准:
- AC1: Given 训练数据 `data/train.csv`,When 执行训练流程,Then 完成数据预处理(缺失值处理、类别编码、特征标准化)。
- AC2: Given 训练流程执行完毕,When 在测试集 `data/test.csv` 上评估,Then AUC ≥ 0.75 且 F1 ≥ 0.6。
- AC3: Given 训练完成,When 保存模型,Then 模型文件和预处理管道(pipeline)序列化为 `models/model.pkl`,可被预测模块加载。
- AC4: Given 训练脚本 `src/model_train.py`,When 以模块方式直接运行,Then 完成训练并打印评估指标;When 作为模块导入,Then 不自动执行训练。
- AC5: Given 模型训练完成后,When 运行模型相关单元测试,Then 测试覆盖:数据加载、预处理、训练、评估、模型保存/加载。

技术备注:
- 使用 scikit-learn 的 `Pipeline` + `ColumnTransformer` 构建端到端预处理+训练管道。
- 特征工程注意:duration 在真实预测场景通常是未知的(通话后才知),模型训练时需保留但预测 UI 中可选是否输入;初版训练时使用 duration,后续可迭代。
- `models/` 目录通过 `.gitignore` 排除,训练产物仅本地使用。

---

### US-4 在线预测系统 · 状态: Backlog

作为 **银行业务人员**,
我想要 在 Web 页面上通过点选表单输入客户信息,
以便 实时得到该客户是否会认购定期存款的预测结果,辅助营销决策。

验收标准:
- AC1: Given 应用已启动且模型已训练,When 访问预测页面,Then 显示输入表单,包含所有可用的客户特征字段。
- AC2: Given 预测页表单,When 类别特征(如 job、marital、education 等),Then 以下拉选择框呈现,选项来自训练数据的合法取值。
- AC3: Given 预测页表单,When 数值特征(如 age、campaign 等),Then 以数字输入框呈现,包含合理的最小/最大值约束。
- AC4: Given 用户已填写所有必填字段,When 点击"预测"按钮,Then 系统加载已训练模型,显示预测结果:是否认购(yes/no)及置信度/概率。
- AC5: Given 用户未填写必填字段,When 点击"预测",Then 提示用户补全所有必填项,不发起预测。
- AC6: Given 模型文件不存在,When 访问预测页面或点击预测,Then 显示友好提示"模型尚未训练,请先执行模型训练"。
- AC7: Given 预测请求,When 从输入到返回结果,Then 响应时间不超过 2 秒。

技术备注:
- 预测逻辑封装在 `src/predict.py`,支持 `load_model()` + `predict(input_data: dict)` 两个接口。
- duration 字段设为可选项(实际业务中事前不知道通话时长),默认值可置为训练集均值或中位数。

---

## 5. 非功能需求

- **安全**:密钥只进 Secrets,不进 Git;本项目的输入均为表单点选,无用户认证需求。
- **可维护**:一需求一小 PR(≤400 行),避免大爆炸式提交。
- **可测试**:核心逻辑必须有单元测试(数据加载、预处理、模型训练、预测推理)。
- **可部署**:本地 Docker 部署;`docker run` 后服务在 `http://localhost:8888` 可达。
- **性能**:数据分析页响应 ≤3 秒,预测接口响应 ≤2 秒。
- **浏览器兼容**:支持 Chrome / Edge / Safari 最新版。
