# 答辩 PPT 大纲（15 页）

按以下 15 页组织答辩演示文稿，每页给出标题、要点与建议插图。图表均可在 `figures/` 目录中找到（由 `notebooks/01_movielens_recommendation.ipynb` 生成）。

## 第 1 页 封面

- 标题：基于协同过滤与矩阵分解的 MovieLens 电影推荐系统
- 姓名 / 学号 / 班级 / 指导教师 / 答辩日期

## 第 2 页 目录 / 汇报结构

- 研究背景 → 数据集与预处理 → 算法设计 → 实验结果 → 深入分析 → 结论与展望

## 第 3 页 研究背景与目标

- 推荐系统在视频/电影平台中的价值
- 本项目目标：在 MovieLens 100K 上系统对比 UserCF / ItemCF / SVD / Hybrid（可选 NeuralCF），从评分预测与 Top-N 排序两个维度评估

## 第 4 页 数据集介绍

- MovieLens 100K：943 用户、1682 部电影、100,000 条评分（1-5 分）
- 数据字段：评分记录、电影元数据（标题/类型/年份/IMDb 链接）

## 第 5 页 数据清洗与探索性分析（EDA）

- 清洗步骤：缺失值处理、年份解析、类型 One-Hot
- 关键发现：评分分布、用户/电影评分数的长尾分布、类型分布

插图：`rating_distribution.png`、`ratings_per_user.png`、`ratings_per_movie.png`、`genre_counts.png`

## 第 6 页 数据划分方法

- 按用户时间序列划分（per-user temporal split）：每位用户的早期评分用于训练，后期评分用于测试
- 为何不用随机划分：避免数据泄漏，更贴近真实推荐场景
- 全局时间划分 / 随机划分作为对照实验

## 第 7 页 算法原理（一）：基线与协同过滤

- 基线模型：GlobalMean / UserMean / ItemMean / Bias / Random / MostPopular
- UserCF：用户-用户相似度 + K 近邻加权
- ItemCF：物品-物品相似度 + K 近邻加权

## 第 8 页 算法原理（二）：矩阵分解与混合模型

- SVD：在偏置基线残差上做 `TruncatedSVD`，学习用户/物品隐因子
- Hybrid：融合用户统计、物品统计、类型特征的回归模型
- （可选）NeuralCF：嵌入 + 多层感知机

## 第 9 页 评估指标体系

- 评分预测：RMSE、MAE
- Top-N 排序：Precision@K、Recall@K、HitRate@K、NDCG@K
- 多样性与新颖性：覆盖率、类型多样性、流行度新颖性

## 第 10 页 模型对比结果：RMSE / MAE

- 各模型 RMSE/MAE 对比图与表格
- 最优模型：SVD（基于偏置残差的 TruncatedSVD）

插图：`model_comparison.png`

## 第 11 页 Top-N 排名评估结果

- Precision@10 / Recall@10 / HitRate@10 / NDCG@10 / 覆盖率对比
- 相关性阈值（`rating >= 4.0`）及其敏感性

插图：`topn_metrics_comparison.png`

## 第 12 页 超参数敏感性分析

- UserCF/ItemCF 的 K 值对 RMSE 的影响
- SVD 隐因子数对 RMSE/MAE 的影响
- 偏置正则化系数的敏感性与最优取值

插图：`cf_k_sensitivity.png`、`svd_components_sensitivity.png`、`bias_regularization_sensitivity.png`

## 第 13 页 深入分析：冷启动、稀疏性与误差案例

- 按用户活跃度 / 电影热度分层的 RMSE
- 按电影类型的 RMSE/MAE 分解
- 数据集稀疏度（93.7%）对协同过滤的影响

插图：`user_tier_rmse.png`、`item_tier_rmse.png`、`genre_rmse.png`

## 第 14 页 系统演示（可选加分项）

- Streamlit / PyQt5 演示系统：个性化推荐结果展示、电影详情页、数据可视化
- 强调：UI 仅作为算法结果的展示与交互入口，核心工作量在算法与评估

插图：`figures/screenshots/`（运行应用后自行截图）

## 第 15 页 结论与展望

- 主要结论：最优 RMSE 模型为 SVD；个性化模型在当前离线 Top-N 协议下排序表现接近流行度基线，原因与改进方向
- 研究局限：数据集规模、隐式反馈缺失、CF 可扩展性
- 未来工作：更精细的候选生成与重排序、引入隐式反馈、更大数据集验证
- 致谢与 Q&A
