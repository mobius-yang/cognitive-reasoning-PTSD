# 认知推理 课程项目

本仓库包含一个从特征表（Excel）出发的建模与评估流水线：

- 输入：
	- `data/task1_scored.xlsx`（写作文本特征/打分结果，逐条或逐次记录）
	- `data/task2_processing_data.xlsx`（量表/生理等特征，通常包含 T1/T2/T3）
- 处理：在“被试（Name）层面”聚合并合并数据，避免 CV 泄漏
- 输出：`results/` 下的指标 `metrics.json` 以及若干图（混淆矩阵、ROC、PR、校准、特征重要性等）

## 环境要求

- Python 版本：建议 3.9+（3.8+ 也通常可用）
- 依赖：见 `requirements.txt`，核心依赖包括 `pandas`、`scikit-learn`、`matplotlib`、`seaborn`、`xgboost`



## 数据准备（请自行完成）

把两份 Excel 放在 `data/` 目录下：

- `data/task1_scored.xlsx`
- `data/task2_processing_data.xlsx`

流水线会自动做一些列名兼容：

- task1 里支持 `name` 或 `Name`（会统一为 `name` 再聚合为 `Name`）
- task2 里支持 `Name` 或 `name`（会统一为 `Name`）

## 运行本代码

最常用：

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

## 输出说明

运行结束后，`results/` 中会生成：

- `metrics.json`：样本量、阈值、分类/回归指标、以及不同模型的 OOF 指标汇总
- 二分类诊断图（按模型前缀命名）：混淆矩阵、ROC、PR、校准曲线
- `feature_importance.png`：分类模型特征重要性（LogReg abs(coef) 或 XGBoost gain）
- `feature_importance_regression.png`：回归随机森林特征重要性
- （如果同时存在 `PCL_T2` 和 `PCL_T3`）额外输出三分类混淆矩阵

## 项目结构

- `src/model.py`：命令行入口（解析参数并调用 pipeline）
- `src/modeling/pipeline.py`：核心建模流程（读取、聚合、建模、评估、画图、写结果）
- `src/modeling/core_utils.py`：流水线使用的通用函数（聚合、CV 选择、阈值选择、I/O 等）
- `src/modeling/plots.py`：画图函数
- `scripts/run_model.sh`：一行命令运行 pipeline

## 复现实验建议

- 固定随机种子：使用 `--seed`
- 确保输入 Excel 的列名与内容一致（特别是 `Name/name`、`PCL_T1`、`PCL_T2/PCL_T3` 等）
