# 认知推理 课程项目

本仓库包含一个从特征表（Excel）出发的建模与评估流水线：

- 输入：
	- `data/task1_scored.xlsx`（写作文本特征/打分结果，逐条或逐次记录）
	- `data/task2_processing_data.xlsx`（量表/生理等特征，通常包含 T1/T2/T3）
- 处理：在“被试（Name）层面”聚合并合并数据，避免 CV 泄漏
- 输出：`results/` 下的指标 `metrics.json` 以及若干图（混淆矩阵、ROC、PR、校准、特征重要性等）

## 环境要求

- Python 版本：建议 3.9+（3.8+ 也通常可用）
- 依赖：见 `requirements.txt`，核心依赖包括 `pandas`、`scikit-learn`、`matplotlib`、`seaborn`、`xgboost`、`statsmodels`


## 数据准备（请自行完成）

把两份 Excel 放在 `data/` 目录下：

- `data/task1_scored.xlsx`
- `data/task2_processing_data.xlsx`

流水线会自动做一些列名兼容：

- task1 里支持 `name` 或 `Name`（会统一为 `name` 再聚合为 `Name`）
- task2 里支持 `Name` 或 `name`（会统一为 `Name`）

## 运行本代码

```bash
python src/model.py \
	--task1 data/task1_scored.xlsx \
	--task2 data/task2_processing_data.xlsx \
	--results-dir results \
	--seed 42
```
或一键式：
```bash
bash scripts/run_model.sh
```

## 建模方法

本流水线实现以下三个建模任务：

### 任务 A：二分类（预测"无反应者" vs "有反应者"）

- 目标变量：
  - `PCL_delta` = 随访时点 PCL (`PCL_T3` 或 `PCL_T2`) - 基线 PCL (`PCL_T1`)
  - `Is_NonResponder` = 1 if `PCL_delta > -10`，否则 0（阈值 -10 表示下降 10 分以上视为"有反应"）

- 特征：
  - 基础文本特征（7 维 × 3 聚合方式 = 21 维）：`VAM_index`、`SAM_index`、`temporal`、`coherence`、`reflection`、`sensory`、`arousal` 的被试层面均值、标准差、线性斜率
  - 高级机制特征（3 维，自动挖掘）：
    - `Advanced_dissociation_index_mean`：主观-客观错位指数均值（NLP arousal vs SUDS 差异的绝对值）
    - `Advanced_dissociation_index_max`：主观-客观错位指数峰值
    - `Advanced_reflection_max`：意义加工突破峰值（reflection 在各 session 的最大值）
  - SUDS 时间序列动力学特征（自动提取）：
    - 基础：`SUDS_after_mean` / `SUDS_after_std` / `SUDS_after_slope` / `SUDS_n_obs`
    - ARIMA：`SUDS_ARIMA_(p,d,q)` + `SUDS_ARIMA_ar1`（惯性/inertia）等
    - GARCH(1,1)：`SUDS_GARCH_(omega,alpha1,beta1)` + 条件方差汇总（如 `SUDS_GARCH_sigma2_mean/max/last`）
  - 基线量表特征：所有以 `_T1` 结尾的列（自动识别，如 `PCL_T1`、`GAD_T1`、`PHQ_T1` 等）
  - 类别型特征（如有）会自动做 one-hot 编码（drop_first=True）
  - 总计：文本特征（基础+高级）+ SUDS 动力学特征 + 基线量表特征（编码后以 `results/metrics.json` 为准）

- 模型（使用 Stratified K-Fold CV + GridSearchCV 调参）：
  1. Logistic Regression（`logreg`）
     - Pipeline: StandardScaler → LogisticRegression (max_iter=2000, class_weight="balanced")
     - 参数网格：`C` ∈ {0.1, 1.0, 10.0}
  2. XGBoost（`xgb`）
     - 参数网格：
       - `n_estimators` ∈ {100, 300}
       - `max_depth` ∈ {3, 5}
       - `learning_rate` ∈ {0.05, 0.1}
       - `subsample` ∈ {0.8, 1.0}
       - `colsample_bytree` ∈ {0.8, 1.0}
     - 自动平衡正负样本权重：`scale_pos_weight = neg_samples / pos_samples`

- 评估指标（OOF = Out-Of-Fold 交叉验证预测）：
  - ROC-AUC：曲线下面积，越接近 1 表示模型区分能力越强
  - AUPRC（平均精确率）：PR 曲线下面积，在不平衡数据中比 AUC 更敏感
  - Brier Score：概率预测误差，越小越好（范围 0-1）
  - Balanced Accuracy：(Sensitivity + Specificity) / 2，适合不平衡数据
  - F1 Score：精确率与召回率的调和平均
  - Threshold：通过最大化 F1 自动选择的最优分类阈值
  - Sensitivity (Recall)：真阳性率 = TP / (TP + FN)
  - Specificity：真阴性率 = TN / (TN + FP)
  - Confusion Matrix：{TN, FP, FN, TP} 四元组

- 最佳模型选择：根据 AUPRC 选择最优模型（因为正负样本可能不平衡，AUPRC 更稳定）

### 任务 B：回归（预测 PCL 症状减少量 `PCL_delta`）

- 目标变量：`PCL_delta`（连续值）
- 模型：Random Forest Regressor (n_estimators=500)
- 评估指标：
  - OOF R²（决定系数）：表示模型解释方差的比例，越接近 1 越好

### 任务 C：三分类（仅当同时存在 `PCL_T2` 和 `PCL_T3` 时才运行）

- 目标变量：`Response_Class`
  - `2`：快速反应者（`PCL_T2 - PCL_T1 ≤ -10`）
  - `1`：慢速反应者（`PCL_T2 - PCL_T1 > -10` 且 `PCL_T3 - PCL_T1 ≤ -10`）
  - `0`：无反应者（两个时点均未达阈值）

- 模型：SVM (RBF kernel, class_weight="balanced")
  - Pipeline: VarianceThreshold → StandardScaler → SVC(probability=True)
  - 参数网格：`C` ∈ {0.1, 1, 10}，`gamma` ∈ {"scale", 0.01, 0.1}
  - 评分指标：`f1_macro`（多分类 F1 的宏平均）

- 评估：生成三分类混淆矩阵并报告分类报告（precision/recall/f1）

### 数据聚合与 CV 策略

- 被试层面聚合：task1 的多条记录按 `name` 分组聚合为均值/标准差/斜率；task2 取每个被试的首条记录
- 避免数据泄漏：合并前删除 task2 中与 task1 重复的列（除 `Name`）；仅使用 `_T1` 基线特征，排除 `_T2`/`_T3` 和衍生的 `_change`/`_mean`/`_std` 列
- 交叉验证：
  - 二分类/三分类：Stratified K-Fold（最多 5 折，根据最小类样本数自动调整）
  - 回归：标准 K-Fold（最多 5 折）

### 无监督分析：谱聚类（Spectral Clustering）

在监督学习之前，流水线会自动执行无监督聚类分析，用于探索数据中的潜在亚群结构：

- 聚类特征选择：优先使用高级机制特征 + 关键动态特征
  - 错位指数（mean & max）
  - 意义加工峰值（reflection_max）
  - 认知特征斜率（VAM_index_calculate_slope, reflection_calculate_slope）
  - 连贯性均值（coherence_mean）
  
- 算法：Spectral Clustering (n_clusters=3, affinity='nearest_neighbors')
  - 适合发现非凸形状的簇
  - 基于图论的谱分解方法

- 可视化输出：
  - `spectral_clustering_pca.png`：PCA 降维后的聚类散点图
  - `spectral_clustering_heatmap.png`：各簇在特征空间的 Z-score 热图

- 应用场景：
  - 探索治疗反应的异质性（快速/慢速/无反应者的潜在子类型）
  - 辅助解释模型预测（聚类标签可作为辅助特征或可视化维度）
  - 发现数据驱动的患者分层模式

## 输出说明

运行结束后，`results/` 目录会自动生成以下文件：

### 1. `metrics.json` —— 完整指标汇总

JSON 格式的所有建模结果，包括：

#### 数据概况
- `n_samples_subject_level`：被试层面样本量（本次：41）
- `followup_col`：使用的随访时点 PCL 列名（如 `PCL_T3`）
- `threshold_delta`：定义有效反应的阈值（-10 表示 PCL 下降 10 分以上视为有反应）
- `class_distribution(Is_NonResponder)`：两类样本分布（0=有反应者，1=无反应者）
  - 示例：`{"0": 29, "1": 12}` 表示 29 人有反应，12 人无反应
- `n_text_features`：写作文本特征总数（7 维基础特征 × 3 聚合方式 + 3 维高级特征 = 24）
- `n_baseline_scale_features`：基线量表特征数（所有 `_T1` 结尾列，如 8 个）
- `n_model_features_after_dummies`：one-hot 编码后最终特征维数（如 32）

#### 二分类结果（`binary_models`）
每个模型（`logreg`、`xgb`）各包含 `oof` 子对象，记录 Out-Of-Fold 交叉验证结果：

- `roc_auc`：ROC 曲线下面积（0.5-1.0，越高越好）
  - 示例：LogReg=0.586，XGB=0.537
- `auprc`：Precision-Recall 曲线下面积（不平衡数据更敏感的指标）
  - 示例：LogReg=0.456，XGB=0.427
- `brier`：Brier Score 概率校准误差（0-1，越低越好）
  - 示例：LogReg=0.248，XGB=0.282
- `balanced_accuracy`：(灵敏度+特异度)/2（0.5-1.0）
  - 示例：LogReg=0.644，XGB=0.629
- `f1`：F1 分数（精确率和召回率的调和平均）
  - 示例：LogReg=0.516，XGB=0.480
- `threshold`：通过最大化 F1 自动选择的最优分类阈值
  - 示例：LogReg=0.480，XGB=0.524
- `sensitivity`：灵敏度（真阳性率，TPR）= TP / (TP + FN)
  - 示例：LogReg=0.667，XGB=0.500
- `specificity`：特异度（真阴性率，TNR）= TN / (TN + FP)
  - 示例：LogReg=0.621，XGB=0.759
- `confusion_matrix`：混淆矩阵四元组
  - `tn`：真阴性（正确预测为有反应者）
  - `fp`：假阳性（错误预测为无反应者）
  - `fn`：假阴性（错误预测为有反应者）
  - `tp`：真阳性（正确预测为无反应者）
  - 示例：LogReg 为 `{tn:18, fp:11, fn:4, tp:8}`

#### 最佳二分类模型
- `binary_best_by_auprc`：根据 AUPRC 选出的最佳模型名称
  - 示例：`"logreg"` 表示 Logistic Regression 表现最优

#### 回归结果
- `regression_oof_r2`：随机森林回归的 OOF R²（决定系数）
  - 示例：0.019（表示模型解释了约 1.9% 的方差，较低可能因样本量小或特征弱）

### 2. 二分类诊断图（每个模型 4 张）

#### Logistic Regression
- `binary_logreg_confusion_matrix.png`：混淆矩阵热图（行=实际类别，列=预测类别）
- `binary_logreg_roc_curve.png`：ROC 曲线（FPR vs TPR），标注 AUC 值
- `binary_logreg_pr_curve.png`：Precision-Recall 曲线，标注 AUPRC 值
- `binary_logreg_calibration.png`：概率校准曲线（预测概率 vs 实际阳性率），对角线=完美校准

#### XGBoost
- `binary_xgb_confusion_matrix.png`
- `binary_xgb_roc_curve.png`
- `binary_xgb_pr_curve.png`
- `binary_xgb_calibration.png`

### 3. 特征重要性图

- `feature_importance.png`：最佳二分类模型的 Top 20 特征重要性
  - LogReg：按绝对系数值 `|coef|` 排序
  - XGBoost：按 gain-based importance 排序
  - 包含高级特征：如 `Advanced_dissociation_index_mean`（错位指数）、`Advanced_reflection_max`（意义峰值）等
  
- `feature_importance_regression.png`：随机森林回归的 Top 20 特征重要性
  - 按 impurity decrease（基尼不纯度减少量）排序

### 4. 无监督聚类可视化（自动生成）

- `spectral_clustering_pca.png`：谱聚类在 PCA 二维空间的可视化
  - 展示被试在降维后的特征空间中的分布
  - 不同颜色/形状代表不同聚类簇
  
- `spectral_clustering_heatmap.png`：各聚类簇的特征轮廓热图
  - 每个聚类在各特征维度上的 Z-score 标准化值
  - 用于理解各聚类簇的特征模式差异

### 5. 三分类混淆矩阵（可选）

- `confusion_matrix_3class.png`：仅当同时存在 `PCL_T2` 和 `PCL_T3` 时生成
  - 行/列标签：
    - `0`：无反应者（两个时点均未达阈值）
    - `1`：慢速反应者（仅 T3 时达阈值）
    - `2`：快速反应者（T2 时已达阈值）

### 示例输出解读

以本次运行为例（41 个被试，PCL_T3 随访，29 个有反应者 vs 12 个无反应者）：

- 最佳模型：Logistic Regression（AUPRC=0.456 > XGB 的 0.427）
- 性能：
  - ROC-AUC=0.586（略优于随机猜测的 0.5，但提升有限）
  - 灵敏度 66.7%（12 个无反应者中正确识别了 8 个）
  - 特异度 62.1%（29 个有反应者中正确识别了 18 个）
  - F1=0.516（精确率与召回率的平衡点）
- 局限：ROC-AUC 和 AUPRC 均较低，可能因为样本量小（n=41）或特征信息不足
- 回归任务：R²=0.019 表示预测 PCL 症状减少量的效果很弱

建议：
- 若需提升性能，可考虑：增加样本量、特征工程（交互项、多项式特征）、尝试其他模型（LightGBM、CatBoost）
- 通过特征重要性图识别关键特征，聚焦高价值变量
- 高级特征表现：`Advanced_reflection_max`（排名第 6）和 `Advanced_dissociation_index_mean`（排名第 7）在 LogReg 中展现出较强的预测能力，提示意义加工峰值和主客观错位指数是重要的机制性特征
- 聚类分析应用：查看 `spectral_clustering_*.png` 了解数据的潜在分层结构，可能揭示不同治疗反应模式的亚群

## 项目结构

```
.
├── README.md                  # 本文档
├── pyproject.toml             # 项目元信息与依赖声明
├── requirements.txt           # pip 依赖列表
├── scripts/
│   └── run_model.sh           # 一键运行脚本
├── src/
│   ├── model.py               # 命令行入口（解析参数并调用 pipeline）
│   ├── feature_extraction.py # （可选）文本特征提取示例脚本
│   └── modeling/
│       ├── pipeline.py        # 核心建模流程（数据聚合、建模、评估、画图）
│       ├── core_utils.py      # 通用函数（聚合、CV、阈值选择、I/O）
│       ├── plots.py           # 画图函数（混淆矩阵、ROC、PR、校准、特征重要性）
│       ├── feature_mining.py  # （可选）特征挖掘与选择工具
│       └── unsupervised.py    # （可选）无监督学习辅助模块
├── data/                      # 输入数据目录（不上传 GitHub）
│   ├── task1_scored.xlsx
│   └── task2_processing_data.xlsx
└── results/                   # 输出目录（自动生成）
    ├── metrics.json
    ├── binary_logreg_*.png
    ├── binary_xgb_*.png
    ├── feature_importance.png
    ├── feature_importance_regression.png
    └── confusion_matrix_3class.png  (optional)
```


## 复现实验建议

- 固定随机种子：使用 `--seed 42`（或其他整数）确保结果可复现
- 检查数据一致性：
  - 确保 task1 和 task2 的 `Name/name` 列能够正确匹配（大小写、空格、编码等）
  - 确认 `PCL_T1`、`PCL_T2/T3` 列存在且为数值型
- 交叉验证折数：若样本量极小（< 10），CV 可能失败，可在 `pipeline.py` 中降低 `max_splits` 或改用 LeaveOneOut
