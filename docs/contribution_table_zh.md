# Git 贡献与小组分工表

本表用于课程报告/答辩中说明各部分工作内容与分工。表格按"项目模块"组织，便于单人项目按阶段填写工作量，或小组项目按成员分配。请将下表中的姓名/学号替换为实际小组成员信息；若为单人完成，可在"负责人"列统一填写本人姓名，"工作量占比"合计 100%。

## 模块分工表

| 模块 | 主要工作内容 | 对应文件 | 负责人（学号 / 姓名） | 工作量占比 |
|------|--------------|----------|------------------------|------------|
| 数据加载与清洗 | 读取 `u.data`/`u.item`/`u.genre`，缺失值处理、年份解析、类型 One-Hot | `src/data_loader.py`、`src/data_cleaning.py`、notebook 第 1-2 章 | Lv Xuan | 100% |
| 探索性数据分析（EDA） | 评分分布、用户/电影评分数分布、类型分布可视化 | `src/visualization.py`、notebook 第 2 章、`figures/rating_distribution.png` 等 | Lv Xuan | 100% |
| 数据划分 | 按用户时间序列划分训练/验证/测试集，及全局时间/随机划分对照 | `src/preprocessing.py`、notebook 第 3、8 章 | Lv Xuan | 100% |
| 基线模型 | 全局均值、用户均值、物品均值、偏置基线、随机、最受欢迎 | `src/baselines.py`、notebook 第 4 章 | Lv Xuan | 100% |
| 协同过滤算法 | UserCF、ItemCF（余弦/Pearson 相似度，K 近邻预测） | `src/user_based_cf.py`、`src/item_based_cf.py`、`src/similarity.py`、notebook 4.1-4.2 | Lv Xuan | 100% |
| 矩阵分解 SVD | 基于偏置残差的 TruncatedSVD 实现 | `src/svd_model.py`、notebook 4.3 | Lv Xuan | 100% |
| 元数据混合模型 | 特征工程与回归模型 | `src/hybrid_model.py`、notebook 4.4 | Lv Xuan | 100% |
| 神经协同过滤（可选） | 基于 PyTorch 的嵌入+MLP 模型 | `src/neural_cf.py`、notebook 4.5 | Lv Xuan | 100% |
| 评估指标 | RMSE、MAE、Precision@K、Recall@K、HitRate@K、NDCG@K、覆盖率 | `src/metrics.py`、notebook 第 5-6 章 | Lv Xuan | 100% |
| 诊断性分析 | 冷启动、稀疏性、用户/电影分层、类型误差、误差案例分析 | `src/analysis.py`、notebook 第 7-8 章 | Lv Xuan | 100% |
| 超参数敏感性分析 | K 近邻数、SVD 隐因子数、偏置正则化系数敏感性 | notebook 第 11 章、`figures/*_sensitivity.png` | Lv Xuan | 100% |
| 结果解读与结论 | 综合结果讨论、局限性与未来工作 | notebook 第 10、12 章 | Lv Xuan | 100% |
| 演示系统（加分项） | Streamlit / PyQt5 推荐结果展示、数据可视化、电影详情页 | `app.py`、`qt_app.py`、`src/recsys_service.py` | Lv Xuan | 100% |
| 报告与答辩材料 | 实验报告结构、答辩 PPT 大纲、分工表 | `docs/report_structure_zh.md`、`docs/defense_ppt_outline_zh.md`、`docs/contribution_table_zh.md` | Lv Xuan | 100% |

## Git 提交记录摘要

```text
529bd4e  2026-05-20  Complete MovieLens recommendation system project
```

> 提示：若为小组项目，建议每位成员以独立 commit 提交自己负责模块的代码，并在提交信息中注明模块名称（例如 `feat(svd): implement TruncatedSVD recommender`），以便 `git log --author=` 自动统计各成员的工作量，并将统计结果填入上表"工作量占比"列。
