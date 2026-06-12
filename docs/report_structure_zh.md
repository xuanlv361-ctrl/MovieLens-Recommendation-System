# 《MovieLens 100K 电影推荐系统》课程报告结构

本结构按照数据挖掘/数据分析课程的标准实验报告格式编写，可直接作为最终 PDF 报告的章节大纲。每章列出应包含的内容、对应的 notebook 章节（`notebooks/01_movielens_recommendation.ipynb`）以及可直接引用的图表（`figures/`）。撰写报告时按顺序展开各小节文字说明，并插入对应图表与结果表格即可。

## 封面

- 课程名称、实验/课程设计题目：《基于协同过滤与矩阵分解的 MovieLens 电影推荐系统》
- 姓名、学号、班级、指导教师、提交日期

## 摘要（约 200-300 字）

- 简述研究问题（电影评分预测与 Top-N 推荐）、数据集（MovieLens 100K，943 用户 / 1682 电影 / 100,000 条评分）、方法（UserCF、ItemCF、SVD、元数据混合模型，可选 NeuralCF）、评估指标（RMSE、MAE、Precision@K、Recall@K、NDCG@K、覆盖率）与主要结论（最优模型及其 RMSE/MAE，Top-N 指标结果，关键发现）。

## 第一章 引言

- 研究背景：推荐系统在电影/视频平台中的应用价值
- 研究目标：对比不同推荐算法在 MovieLens 100K 上的预测精度与排序质量
- 报告结构说明

对应 notebook：`## 1. Setup`

## 第二章 数据集与探索性数据分析（EDA）

- 数据集来源与字段说明（`u.data`、`u.item`、`u.genre`）
- 数据清洗步骤：缺失值、重复值、类型转换、年份解析
- 评分分布、用户活跃度分布、电影热度分布、电影类型分布

插入图表：
- `figures/rating_distribution.png` — 评分分布
- `figures/ratings_per_user.png` — 用户评分数分布
- `figures/ratings_per_movie.png` — 电影评分数分布
- `figures/genre_counts.png` — 电影类型分布

对应 notebook：`## 2. Exploratory Data Analysis (EDA)`

## 第三章 数据划分与特征工程

- 训练/验证/测试划分方法：**按用户的时间序列划分**（per-user temporal split），说明为何采用该方法以避免数据泄漏
- 全局时间划分与随机划分作为对照（第八章会用到）
- 元数据混合模型的特征工程：用户统计特征、物品统计特征、类型 One-Hot、上映年份、交互特征

对应 notebook：`## 3. Preprocessing - Train / Test Split`，`### 4.4.1 Metadata Hybrid Feature Documentation`

## 第四章 推荐算法设计

逐一介绍每个模型的原理、输入输出与关键超参数：

1. **基线模型**：全局均值、用户均值、物品均值、正则化用户-物品偏置（Bias）、随机评分、最受欢迎（Most Popular）
2. **基于用户的协同过滤（UserCF）**：用户-用户相似度（余弦/Pearson）、K 近邻加权预测
3. **基于物品的协同过滤（ItemCF）**：物品-物品相似度、加权预测
4. **矩阵分解（SVD）**：在偏置基线残差上应用 `TruncatedSVD`，隐因子数 `n_components`
5. **元数据混合模型（Hybrid）**：基于用户/物品统计特征与类型特征的回归模型
6. **（可选）神经协同过滤（NeuralCF）**：基于 PyTorch 的嵌入+MLP 模型

对应 notebook：`## 4. Model Training & Evaluation`（4.1-4.5）

## 第五章 实验结果与模型评估

### 5.1 评分预测指标对比

- 各模型 RMSE / MAE 对比表与柱状图

插入图表：`figures/model_comparison.png`

### 5.2 预测值与残差分析

插入图表（按需选取代表性模型）：
- `figures/pred_vs_actual_*.png` — 预测值 vs 真实值散点图
- `figures/residuals_*.png` — 残差分布图

### 5.3 Top-N 排名评估

- Precision@10、Recall@10、HitRate@10、NDCG@10、Catalog Coverage 对比表
- 关于相关性阈值（`rating >= 4.0`）的说明与敏感性分析（阈值 3.0 / 3.5 / 4.0）

插入图表：
- `figures/topn_metrics_comparison.png`
- `figures/most_popular_recommendation_frequency.png`

### 5.4 最终决策表

- 综合 RMSE/MAE、Top-N 指标、覆盖率、训练/预测耗时、内存占用的最终模型对比表

对应 notebook：`## 5. Results Summary`、`## 6. Top-K Ranking Evaluation`、`## 6.1 Final Decision Table`

## 第六章 超参数敏感性分析

- UserCF/ItemCF 的 `K_NEIGHBORS`（5-50）对 RMSE 的影响
- SVD `n_components` 对 RMSE/MAE 的影响
- 偏置基线正则化系数 `reg` 的敏感性分析与最优取值

插入图表：
- `figures/cf_k_sensitivity.png`
- `figures/svd_components_sensitivity.png`
- `figures/bias_regularization_sensitivity.png`

对应 notebook：`## 11. Hyperparameter Sensitivity`

## 第七章 冷启动、稀疏性与多样性分析

- 数据集稀疏度（93.7% 缺失）对协同过滤的影响
- 按用户活跃度分层的 RMSE（冷启动用户 vs 活跃用户）
- 按电影热度分层的 RMSE（冷门电影 vs 热门电影）
- Top-N 推荐结果的多样性（类型分布）与新颖性（基于流行度的自信息）

插入图表：
- `figures/user_tier_rmse.png`
- `figures/item_tier_rmse.png`
- `figures/diversity_novelty.png`

对应 notebook：`## 7. Diagnostic Error Analysis`

## 第八章 误差案例分析与划分有效性检验

- 按真实评分等级的平均误差分析（误差是否在极端评分上偏大）
- 按电影类型的 RMSE/MAE 分解，找出误差最大/最小的类型
- 全局时间划分、随机划分与按用户时间划分的对比，验证主划分方法的合理性
- K 折交叉验证鲁棒性检验、学习曲线

插入图表：
- `figures/error_by_rating_item_cf.png`
- `figures/genre_rmse.png`
- `figures/learning_curve.png`

对应 notebook：`## 7. Diagnostic Error Analysis`、`## 8. Split Validity Check`、`## 9. K-Fold Robustness, Learning Curves, and Scalability`

## 第九章 结果讨论与结论

- 各模型综合表现总结：最优 RMSE/MAE 模型、最优 Top-N 模型
- 对"个性化模型在当前离线评估协议下排序表现不及流行度基线"现象的解释（稀疏性、阈值、留出集大小等）
- 研究局限性：数据集规模与时效性、隐式反馈缺失、协同过滤的可扩展性
- 未来工作方向：更细粒度的候选生成与重排序、引入隐式反馈、更大规模数据集验证

对应 notebook：`## 10. Result Interpretation`、`## 12. Conclusion`

## 参考文献

- F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets: History and Context. ACM TiiS 2015.
- Sarwar, B. et al. Item-based collaborative filtering recommendation algorithms. WWW 2001.
- Koren, Y. et al. Matrix factorization techniques for recommender systems. Computer 2009.

## 附录（可选加分项）

- 系统演示：Streamlit / PyQt5 推荐结果展示与数据可视化界面截图（详见 [README.md](../README.md) 的"系统架构"与"截图"部分）
- 代码仓库结构说明
- 完整运行环境与复现步骤
