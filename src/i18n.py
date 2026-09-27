"""Bilingual (Simplified Chinese / English) UI strings for the demo apps.

Centralizing the common navigation labels, buttons, and status messages here
keeps the front ends consistent and makes wording changes a one-line edit
instead of a find-and-replace across the apps.

Language model
--------------
- ``_ZH`` is the canonical (and complete) Simplified Chinese string table.
- ``_EN`` holds the English translation for every key in ``_ZH``.
- ``_EN_LITERALS`` maps raw Chinese literals (used as ad-hoc ``t()`` keys in
  ``app.py``) to English, so hardcoded strings can be internationalized by
  simply wrapping them with ``t("中文文案")`` — under Chinese the key itself
  is returned, under English the mapped translation.
- ``set_lang()`` / ``get_lang()`` control the process-wide display language.
  The Streamlit app syncs this from ``st.session_state`` on every rerun; the
  PyQt app never calls it and therefore stays in the default Chinese.

Algorithm/metric names (UserCF, ItemCF, SVD, NeuralCF, Hybrid, RMSE, MAE,
NDCG, ...) are intentionally kept in English/abbreviated form in both
languages, matching common usage in technical writing.
"""

from __future__ import annotations

SUPPORTED_LANGUAGES: dict[str, str] = {"zh": "中文", "en": "English"}
DEFAULT_LANGUAGE = "zh"

_CURRENT_LANG = DEFAULT_LANGUAGE


def get_lang() -> str:
    """Return the current UI language code ("zh" or "en")."""
    return _CURRENT_LANG


def set_lang(lang: str) -> None:
    """Set the current UI language; unknown codes fall back to Chinese."""
    global _CURRENT_LANG
    _CURRENT_LANG = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


_ZH: dict[str, str] = {
    # Navigation
    "nav_home": "首页",
    "nav_for_you": "为你推荐",
    "nav_catalog": "电影库",
    "nav_user_management": "用户管理",
    "nav_rating_records": "评分记录",
    "nav_algorithm_config": "算法配置",
    "nav_model_evaluation": "模型评估",
    "nav_system_stats": "系统统计",
    "nav_admin": "管理员后台",
    "nav_visualization": "数据可视化",
    "nav_similar_movies": "相似电影",
    "entrance_select": "入口选择",
    "entrance_user": "用户入口",
    "entrance_admin": "管理员入口",
    # Buttons
    "btn_search": "搜索",
    "btn_generate_recommendations": "生成推荐",
    "btn_add_movie": "添加电影",
    "btn_edit_movie": "编辑电影",
    "btn_delete_movie": "删除电影",
    "btn_apply_change": "应用更改",
    "btn_save_config": "保存配置",
    "btn_refresh_data": "刷新数据",
    "btn_view_intro": "查看简介",
    "btn_open_imdb": "打开 IMDb 页面",
    "btn_run_evaluation": "运行评估",
    # Common labels
    "label_user_id": "用户ID",
    "label_movie_id": "电影ID",
    "label_title": "标题",
    "label_genre": "类型",
    "label_all": "全部",
    "label_algorithm": "算法",
    "label_neighbors_k": "近邻数 K",
    "label_svd_factors": "SVD 因子数",
    "label_similarity_method": "相似度计算方法",
    "label_recommendation_count": "推荐数量",
    "label_rank_by": "排序方式",
    "label_show_top": "显示前N项",
    "label_username": "用户名",
    "label_password": "密码",
    # Status / messages
    "msg_admin_verified": "管理员身份验证成功",
    "msg_no_candidates": "未找到该用户尚未观看的候选电影。",
    "msg_not_enough_overlap": "重叠评分数据不足，无法推荐相似电影。",
    # Landing page
    "landing_title": "FilmTrace",
    "landing_subtitle": "智能电影推荐平台",
    "landing_description": (
        "基于 MovieLens 100K 数据集构建的智能电影推荐系统，"
        "融合 UserCF、ItemCF、SVD、NeuralCF 与数据分析能力，帮助用户发现感兴趣的电影。"
    ),
    "landing_user_button": "进入用户系统",
    "landing_admin_button": "进入管理员后台",
    "btn_back_to_landing": "返回首页",
    # Auth
    "auth_login_tab": "登录",
    "auth_register_tab": "注册",
    "label_email": "邮箱",
    "label_confirm_password": "确认密码",
    "btn_login": "登录",
    "btn_register": "注册",
    "btn_logout": "退出登录",
    "msg_login_failed": "用户名/邮箱或密码错误，或账号已被禁用。",
    "msg_register_success": "注册成功，正在为你进入新手引导...",
    "msg_password_mismatch": "两次输入的密码不一致。",
    # User-system navigation
    "nav_user_home": "🏠 首页",
    "nav_user_for_you": "🎬 为你推荐",
    "nav_user_catalog": "📚 电影库",
    "nav_user_favorites": "❤️ 我的收藏",
    "nav_user_ratings": "⭐ 我的评分",
    "nav_user_profile_analytics": "📊 我的画像",
    "nav_user_account": "👤 个人中心",
    "sidebar_section_discover": "发现",
    "sidebar_section_personal": "个人",
    # Admin-system navigation
    "nav_admin_stats": "📊 数据统计中心",
    "nav_admin_movies": "🎬 电影管理",
    "nav_admin_users": "👥 用户管理",
    "nav_admin_models": "🧠 推荐模型管理",
    "nav_admin_audit": "📝 操作日志",
    "nav_admin_monitor": "⚙️ 系统监控",
    # Registration onboarding (genre preferences)
    "label_onboarding_genres": "请选择你喜欢的电影类型（可多选）",
    "msg_onboarding_genres_required": "请至少选择一种喜欢的电影类型。",
    "label_user_preferred_genres": "你的偏好类型",
    "recommendation_source_cf": "基于你的历史评分和协同过滤模型生成推荐。",
    "recommendation_source_genre": "基于你注册时选择的电影类型进行冷启动推荐。",
    "recommendation_source_popular": "暂无足够的个性化数据，为你展示全站热门高分电影。",
    "label_recommendation_source": "推荐方式",
    "profile_preferred_genres": "用户偏好类型",
    "profile_preferred_genre_movie_count": "偏好类型电影数量",
    "profile_recommendation_source": "推荐来源",
    "profile_source_registration": "注册偏好",
    "profile_source_history": "历史评分",
    "profile_source_cf": "协同过滤",
    # Language switcher
    "label_language": "语言 / Language",
}


_EN: dict[str, str] = {
    # Navigation
    "nav_home": "Home",
    "nav_for_you": "For You",
    "nav_catalog": "Movie Catalog",
    "nav_user_management": "User Management",
    "nav_rating_records": "Rating Records",
    "nav_algorithm_config": "Algorithm Config",
    "nav_model_evaluation": "Model Evaluation",
    "nav_system_stats": "System Stats",
    "nav_admin": "Admin Console",
    "nav_visualization": "Data Visualization",
    "nav_similar_movies": "Similar Movies",
    "entrance_select": "Select Entrance",
    "entrance_user": "User Entrance",
    "entrance_admin": "Admin Entrance",
    # Buttons
    "btn_search": "Search",
    "btn_generate_recommendations": "Generate Recommendations",
    "btn_add_movie": "Add Movie",
    "btn_edit_movie": "Edit Movie",
    "btn_delete_movie": "Delete Movie",
    "btn_apply_change": "Apply Changes",
    "btn_save_config": "Save Config",
    "btn_refresh_data": "Refresh Data",
    "btn_view_intro": "View Intro",
    "btn_open_imdb": "Open IMDb Page",
    "btn_run_evaluation": "Run Evaluation",
    # Common labels
    "label_user_id": "User ID",
    "label_movie_id": "Movie ID",
    "label_title": "Title",
    "label_genre": "Genre",
    "label_all": "All",
    "label_algorithm": "Algorithm",
    "label_neighbors_k": "Neighbors K",
    "label_svd_factors": "SVD Factors",
    "label_similarity_method": "Similarity Metric",
    "label_recommendation_count": "Recommendations",
    "label_rank_by": "Sort By",
    "label_show_top": "Show Top N",
    "label_username": "Username",
    "label_password": "Password",
    # Status / messages
    "msg_admin_verified": "Administrator identity verified.",
    "msg_no_candidates": "No candidate movies left that this user has not watched.",
    "msg_not_enough_overlap": "Not enough overlapping ratings to recommend similar movies.",
    # Landing page
    "landing_title": "FilmTrace",
    "landing_subtitle": "Intelligent Movie Recommendation Platform",
    "landing_description": (
        "An intelligent movie recommendation system built on the MovieLens 100K dataset, "
        "combining UserCF, ItemCF, SVD, NeuralCF and data analytics to help you discover "
        "movies you will love."
    ),
    "landing_user_button": "Enter User System",
    "landing_admin_button": "Enter Admin Console",
    "btn_back_to_landing": "Back to Home",
    # Auth
    "auth_login_tab": "Sign In",
    "auth_register_tab": "Sign Up",
    "label_email": "Email",
    "label_confirm_password": "Confirm Password",
    "btn_login": "Sign In",
    "btn_register": "Sign Up",
    "btn_logout": "Log Out",
    "msg_login_failed": "Incorrect username/email or password, or the account has been disabled.",
    "msg_register_success": "Registration successful! Taking you to onboarding...",
    "msg_password_mismatch": "The two passwords do not match.",
    # User-system navigation
    "nav_user_home": "🏠 Home",
    "nav_user_for_you": "🎬 For You",
    "nav_user_catalog": "📚 Catalog",
    "nav_user_favorites": "❤️ Favorites",
    "nav_user_ratings": "⭐ My Ratings",
    "nav_user_profile_analytics": "📊 My Profile",
    "nav_user_account": "👤 Account",
    "sidebar_section_discover": "Discover",
    "sidebar_section_personal": "Personal",
    # Admin-system navigation
    "nav_admin_stats": "📊 Stats Center",
    "nav_admin_movies": "🎬 Movie Management",
    "nav_admin_users": "👥 User Management",
    "nav_admin_models": "🧠 Model Management",
    "nav_admin_audit": "📝 Audit Log",
    "nav_admin_monitor": "⚙️ System Monitor",
    # Registration onboarding (genre preferences)
    "label_onboarding_genres": "Pick your favorite movie genres (multiple allowed)",
    "msg_onboarding_genres_required": "Please select at least one favorite genre.",
    "label_user_preferred_genres": "Your Preferred Genres",
    "recommendation_source_cf": "Recommendations generated from your rating history with collaborative filtering.",
    "recommendation_source_genre": "Cold-start recommendations based on the genres you picked at registration.",
    "recommendation_source_popular": "Not enough personal data yet — showing popular top-rated movies.",
    "label_recommendation_source": "Recommendation Source",
    "profile_preferred_genres": "Preferred Genres",
    "profile_preferred_genre_movie_count": "Movies in Preferred Genres",
    "profile_recommendation_source": "Recommendation Source",
    "profile_source_registration": "Registration Preferences",
    "profile_source_history": "Rating History",
    "profile_source_cf": "Collaborative Filtering",
    # Language switcher
    "label_language": "语言 / Language",
}


# English translations for raw Chinese literals used as ad-hoc `t()` keys.
# Under Chinese the literal key itself is displayed; under English the value
# below is shown. Keep entries sorted by the Chinese key for maintainability.
_EN_LITERALS: dict[str, str] = {
    # Table headers / chart labels used as DataFrame column names
    "中文片名": "Title",
    "原片名": "Original Title",
    "加权得分": "Weighted Score",
    "偏好强度": "Preference Strength",
    "用户评分": "User Rating",
    "评分日期": "Rated At",
    "收藏数": "Favorites",
    "最近活动日期": "Last Activity",
    "状态": "Status",
    "活跃等级": "Activity Tier",
    "排名": "Rank",
    "预测评分": "Predicted Score",
    " / 新增海报后点击“刷新海报库”即可更新显示。": " / Click \"Refresh Poster Library\" after adding new posters to update.",
    " 位。": " in the current similar-movie recommendation list.",
    " 偏好编辑": " Edit Preferences",
    " 已从数据库删除。": " has been deleted from the database.",
    " 已存在。": " already exists.",
    " 已添加并保存到数据库。": " has been added and saved to the database.",
    " 条。": ".",
    " 条记录，最多显示前 ": " records, showing at most ",
    " 条评分，平均 ": " ratings, averaging ",
    " 标题已更新并保存到数据库。": " title has been updated and saved to the database.",
    " 的信息已更新并保存到本地数据库。": " has been updated and saved to the local database.",
    " 的历史评分，系统已自动生成以下个性化推荐结果。": "'s rating history, the system has generated the following personalized recommendations.",
    " 秒": "s",
    " 编辑": " Edit",
    " 部。": ".",
    " 部电影。": " most similar movies.",
    " 部电影，显示前 ": " movies, showing the first ",
    "#### 算法说明": "#### Algorithm Overview",
    "#### 管理员操作日志": "#### Admin Audit Log",
    "**NeuralCF（神经协同过滤）** ": "**NeuralCF (Neural Collaborative Filtering)** ",
    "FilmTrace | 电影推荐系统 Movie Recommendation": "FilmTrace | Movie Recommendation System",
    "MAE / RMSE（数值越低越好）": "MAE / RMSE (lower is better)",
    "MovieLens 100K 中综合评分最高、最受欢迎的电影，海报优先使用本地图库。": "The highest-rated and most popular movies in MovieLens 100K; posters come from the local library first.",
    "MovieLens 100K 提供了元数据和 IMDb 链接，但不包含海报图片或剧情简介。": "MovieLens 100K provides metadata and IMDb links, but no posters or plot summaries.",
    "MovieLens 100K 数据集记录了用户的显式评分及对应的 Unix 时间戳。": "The MovieLens 100K dataset records explicit user ratings with Unix timestamps.",
    "MovieLens 100K 数据集评分时间主要集中在 1997-1998 年，因此日期为历史数据，不代表当前时间。": "MovieLens 100K ratings are mostly from 1997-1998, so the dates are historical and do not reflect the current time.",
    "MovieLens用户-": "MovieLens User-",
    "Precision@10 / Recall@10 / HitRate@10 / NDCG@10（数值越高越好）": "Precision@10 / Recall@10 / HitRate@10 / NDCG@10 (higher is better)",
    "Top-10 排序质量对比": "Top-10 Ranking Quality Comparison",
    "♡ 收藏": "♡ Favorite",
    "♥ 已收藏": "♥ Favorited",
    "⭐ 评价": "⭐ Rate",
    "　平均评分 ": " · Avg ",
    "　评分数 ": " · Ratings ",
    "。": " not found.",
    "》（电影": "\" (Movie ",
    "」。": ".",
    "上映年代分布": "Release Decade Distribution",
    "上映年份": "Release Year",
    "上榜电影数": "Movies Listed",
    "与你收藏的电影类型相似（{genres}）": "Shares genres with your favorites ({genres})",
    "与相似用户的兴趣模式（协同过滤 UserCF / ItemCF / SVD 及深度模型 NeuralCF），": "and the taste patterns of similar users (collaborative filtering UserCF / ItemCF / SVD plus the NeuralCF deep model), ",
    "严格评分型用户": "Harsh Rater",
    "个人中心": "Account",
    "中偏好": "Medium",
    "为什么这样推荐？": "Why These Recommendations?",
    "为你推荐": "For You",
    "人均评分数": "Avg Ratings per User",
    "从{source}来看，系统更倾向于为你推荐{top1}、{others}等类型电影，": "Based on {source}, the system tends to recommend genres such as {top1} and {others}, ",
    "从{source}来看，系统更倾向于为你推荐{top1}类电影，": "Based on {source}, the system tends to recommend {top1} movies, ",
    "代表性高分电影": "Representative High-Rated Movies",
    "以下为各算法在按用户时间序列划分的留出测试集上的评分预测误差（数值越低越好）。": "Rating prediction errors of each algorithm on a per-user time-based holdout test set (lower is better).",
    "优先推荐与你偏好类型相近、评分表现较高、且与相似用户口味一致的电影。": "prioritizing movies close to your preferred genres, with strong ratings, and matching similar users' tastes.",
    "会分析你评分较高的电影，找到与这些电影评分模式相似的其他电影": "analyzes the movies you rated highly, finds other movies with similar rating patterns ",
    "会找到与你评分习惯相似的其他用户（基于用户-用户相似度），": "finds other users with rating habits similar to yours (user-user similarity), ",
    "位用户": "users",
    "低偏好": "Low",
    "低活跃": "Low Activity",
    "你之前已评价过这部电影，可在下方修改。": "You have rated this movie before; you can update it below.",
    "你历史评分中各电影类型出现的次数（按评分次数排序）。": "How often each genre appears in your rating history (sorted by count).",
    "你在注册时选择的偏好类型，及片库中对应的电影数量。": "Genres you picked at registration and the number of matching movies in the catalog.",
    "你的电影兴趣标签": "Your Movie Taste Tags",
    "你的评分": "Your Rating",
    "你给出的评分（1-5 星）的次数分布。": "Distribution of the ratings (1-5 stars) you gave.",
    "你评分过的电影按上映年代统计，反映你常关注的电影年代。": "Movies you rated grouped by release decade — the eras you watch most.",
    "你还没有收藏电影，收藏更多电影后可以生成更准确的相似电影推荐。": "No favorites yet. Add favorites to get better similar-movie recommendations.",
    "你还没有设置偏好类型，前往“我的画像”页面注册偏好后可获取个性化推荐。": "You have not set preferred genres yet. Visit \"My Profile\" to set them and get personalized recommendations.",
    "你还没有足够的评分记录，完成更多电影评分后，": "You don't have enough ratings yet. ",
    "保存修改": "Save Changes",
    "保存偏好设置": "Save Preferences",
    "修改密码": "Change Password",
    "修改将保存到本地数据库，不会覆盖原始 MovieLens 数据文件。": "Changes are saved to the local database and will not overwrite the original MovieLens data files.",
    "修改评价": "Edit Review",
    "偏好冷启动适配": "Cold-Start Ready",
    "偏好年代": "Preferred Era",
    "偏好数据不足": "Not enough preference data",
    "偏好电影类型": "Preferred Genres",
    "偏好类型": "Preferred Genres",
    "偏好类型分析": "Genre Preference Analysis",
    "偏好高分类型": "Top-Rated Genre",
    "全站热门高分电影推荐": "Popular top-rated movies across the site",
    "全部用户在 MovieLens 100K 数据集上的评分取值分布。": "Distribution of all user ratings in the MovieLens 100K dataset.",
    "共 ": "Total ",
    "共找到 ": "Found ",
    "其中{top1}类电影占比最高，说明你的评分历史和偏好类型与该类电影具有较高匹配度。": "with {top1} taking the largest share — your rating history and preferred genres match this genre well.",
    "写下你对这部电影的看法……": "Share your thoughts on this movie...",
    "冷启动分析": "Cold-Start Analysis",
    "分析新用户 / 新电影在评分数据不足时的推荐表现。": "Analyzes recommendation performance for new users / movies with sparse rating data.",
    "创建 FilmTrace 账号": "Create Your FilmTrace Account",
    "剧情片关注者": "Drama Watcher",
    "动作片爱好者": "Action Fan",
    "动画偏好": "Animation Fan",
    "协同过滤推荐适配": "CF-Ready",
    "历史数据用户": "Legacy Dataset User",
    "历史评分行为（共 ": "your rating history (",
    "原片名：": "Original title: ",
    "取消": "Cancel",
    "可视化对比": "Visual Comparison",
    "各指标归一化后的横向对比（数值越大越好）": "Side-by-side comparison of normalized metrics (higher is better)",
    "各类型平均评分与评分数量": "Average Rating and Rating Count by Genre",
    "各类型电影数量（可重复计算多类型电影）": "Number of movies per genre (multi-genre movies counted in each)",
    "各评分等级（1～5星）的评分数量分布": "Distribution of ratings across the 1-5 star scale",
    "喜剧片爱好者": "Comedy Fan",
    "因为你喜欢「{genres}」": "Because you like \"{genres}\"",
    "在当前相似电影推荐列表中排名第 ": "Ranked #",
    "在按用户时间序列划分的留出测试集上评估各算法的评分预测误差与 Top-10 排序质量": "Evaluates the rating prediction error and Top-10 ranking quality of each algorithm on a per-user time-based holdout set ",
    "基于 MovieLens 100K 数据集构建": "Built on the MovieLens 100K dataset",
    "基于 MovieLens 100K 评分数据、电影类型与推荐结果生成的可视化分析，帮助理解用户行为与推荐结果分布。": "Visualizations built from MovieLens 100K ratings, genres, and recommendation results to help understand user behavior and recommendation distributions.",
    "基于 PyTorch 构建，将用户与电影映射为嵌入向量并通过多层感知机（MLP）": "built on PyTorch, maps users and movies to embedding vectors and uses a multi-layer perceptron (MLP) ",
    "基于 PyTorch 的神经协同过滤，用嵌入向量与多层感知机建模用户-电影非线性交互。": "PyTorch-based neural collaborative filtering that models nonlinear user-movie interactions with embeddings and an MLP.",
    "基于 PyTorch 的神经协同过滤，用用户/电影嵌入向量与多层感知机（MLP）建模非线性交互，经端到端训练预测评分。": "PyTorch-based neural collaborative filtering: user/movie embeddings plus an MLP model nonlinear interactions, trained end-to-end to predict ratings.",
    "基于 RMSE / MAE / Precision@K / Recall@K / NDCG@K 等指标系统评估各算法。": "Systematic evaluation of each algorithm with RMSE / MAE / Precision@K / Recall@K / NDCG@K metrics.",
    "基于你喜欢的类型推荐": "Recommended from Your Favorite Genres",
    "基于你的收藏相似推荐": "Similar to Your Favorites",
    "基于你的评分历史、偏好类型与推荐结果生成的个性化电影兴趣分析。": "A personalized analysis of your movie taste based on your rating history, preferred genres, and recommendations.",
    "基于物品-物品相似度，查找与目标电影评分模式相近的其他电影。": "Finds movies with similar rating patterns using item-item similarity.",
    "基于用户 ": "Based on user ",
    "基于相似用户": "User-Based",
    "基于相似用户的历史评分，为目标用户推荐其“同好”喜欢的电影。": "Recommends movies liked by like-minded users based on similar users' rating history.",
    "基于相似电影": "Item-Based",
    "学习两者之间的非线性交互，经端到端训练后预测你对未观看电影的评分，再按预测评分排序推荐。": "to learn nonlinear interactions between them, trained end-to-end to predict your ratings for unwatched movies, ranked by predicted rating.",
    "密码已更新，下次登录请使用新密码。": "Password updated. Use the new password next time you sign in.",
    "对所选电影评分较高的用户对该电影也表现出相似的评分模式。": "users who rated the selected movie highly show similar rating patterns for this one.",
    "将用户-电影评分矩阵分解为隐因子矩阵，通过隐因子向量的内积预测缺失评分。": "Factorizes the user-movie rating matrix into latent factors and predicts missing ratings via inner products.",
    "尚未设置的偏好类型": "no preferred genres set",
    "已收藏": "Favorited",
    "已登录": "Logged in",
    "已禁用": "Disabled",
    "已评分电影数": "Rated Movies",
    "帮助分析你的真实观影偏好。": "to reveal your real taste in movies.",
    "平均评分": "Avg Rating",
    "平均评分 ": "Avg ",
    "平均评分 ≥ 4.0": "Avg rating ≥ 4.0",
    "年代": "Decade",
    "并将这些相似用户喜欢但你还未观看的电影推荐给你。": "and recommends movies those similar users liked but you have not watched yet.",
    "并用这些隐因子计算你对未观看电影的预测评分，再按预测评分排序推荐。": "and uses these factors to predict your ratings for unwatched movies, ranked by predicted rating.",
    "异常": "Error",
    "异常评分数量": "Invalid Ratings",
    "当前为你匹配的推荐方式为「": "Your current recommendation source: ",
    "当前密码": "Current Password",
    "当前密码不正确。": "Current password is incorrect.",
    "当前推荐结果为空，以全站热门电影的类型分布作为参考。": "No recommendations yet — showing the genre distribution of popular movies instead.",
    "当前数据不足，完成更多评分或推荐后可生成更完整的可视化分析。": "Not enough data yet. Rate more movies or generate recommendations for a fuller analysis.",
    "当前显示热门高分电影 Top 20，输入关键词或选择类型可进一步筛选。": "Showing the top 20 popular movies. Enter keywords or pick a genre to filter further.",
    "当前没有注册账号，无法编辑偏好类型。": "No registered accounts available for preference editing.",
    "当前电影": "Current movie ",
    "当前算法配置已保存到会话状态。": "Algorithm configuration saved to session state.",
    "当前评分数据较少，继续评分后可生成更完整的电影画像。": "Limited rating data so far — keep rating to build a fuller movie profile.",
    "当前默认算法：": "Current default algorithm: ",
    "恐怖片探索者": "Horror Explorer",
    "惊悚片爱好者": "Thriller Fan",
    "我的平均评分": "My Avg Rating",
    "我的收藏": "My Favorites",
    "我的电影画像": "My Movie Profile",
    "我的画像": "My Profile",
    "我的评分": "My Ratings",
    "我的评分分布": "My Rating Distribution",
    "我的评分数": "My Ratings",
    "我的评分行为": "My Rating Behavior",
    "我的评分行为趋势": "My Rating Activity Trend",
    "我给出的评分（1-5 星）次数分布。": "Distribution of the ratings (1-5 stars) I gave.",
    "找到与目标用户兴趣相似的其他用户，根据这些用户的评分加权预测目标用户对电影的偏好。": "Finds users with similar tastes and predicts the target user's preferences from their weighted ratings.",
    "按月统计的我的评分活动次数。": "My monthly rating activity.",
    "按用户评分数量划分的活跃区间人数分布": "Users grouped into activity buckets by rating count",
    "按电影查找相似电影": "Find Similar Movies by Movie",
    "推荐来源": "Recommendation Source",
    "推荐理由：": "Reason: ",
    "推荐生成耗时：": "Recommendation time: ",
    "推荐电影数": "Recommended Movies",
    "推荐系统介绍": "About the Recommender System",
    "推荐结果": "your recommendations",
    "推荐结果类型分布": "Genre Distribution of Recommendations",
    "推荐结果解读": "Reading Your Recommendations",
    "推荐评估": "Recommendation Evaluation",
    "推荐该电影的原因：在 MovieLens 100K 评分矩阵中，": "Why this movie: in the MovieLens 100K rating matrix, ",
    "推荐说明": "How It Works",
    "提交": "Submit",
    "搜索电影标题或电影ID": "Search by movie title or ID",
    "操作": "Action",
    "支持按电影ID、电影名称和类型筛选 MovieLens 电影数据。": "Filter MovieLens movies by ID, title, and genre.",
    "收藏": "Favorite",
    "收藏总数": "Total Favorites",
    "收藏成功": "Added to favorites",
    "数据分布概览": "Data Distribution Overview",
    "数据可视化": "Data Visualization",
    "数据可视化分析": "Visual Analytics",
    "数据库状态": "Database Status",
    "数据稀疏度": "Data Sparsity",
    "数据集概览": "Dataset Overview",
    "数据集规模、用户行为与内容结构的全局视图。": "A global view of dataset scale, user behavior, and content structure.",
    "数量": "Count",
    "新密码": "New Password",
    "新密码不能为空。": "New password cannot be empty.",
    "新注册用户数": "Registered Accounts",
    "无": "None",
    "显示标题": "Display Title",
    "普通": "Regular",
    "暂无": "N/A",
    "暂无与你收藏的电影类型相似的新电影，去探索更多电影吧。": "No new movies with genres similar to your favorites — go explore more.",
    "暂无偏好": "No preferences",
    "暂无收藏，请在下方搜索并添加喜欢的电影。": "No favorites yet. Search below to add movies you like.",
    "暂无数据": "No data",
    "暂无数据。": "No data available.",
    "暂无法从你的收藏中识别出电影类型，去收藏更多电影试试。": "Could not identify genres from your favorites — try adding more.",
    "暂无热门电影数据。": "No popular movie data available.",
    "暂无符合你偏好类型的新电影，去探索更多电影吧。": "No new movies matching your preferred genres — go explore more movies.",
    "暂无评价内容": "No review yet",
    "暂无评分": "No rating",
    "暂无评分数据。": "No rating data available.",
    "暂无评分数据，评分相关画像（类型评分分布/评分行为分析/代表性电影）将在你评分后生成。": "No rating data yet. Rating-based profile sections (genre distribution / behavior analysis / top-rated movies) will appear after you rate movies.",
    "暂无评分记录，去“浏览与搜索”页面为喜欢的电影打分吧。": "No ratings yet. Go to \"Browse & Search\" to rate movies you like.",
    "暂无评分记录，去“电影库”页面为喜欢的电影打分吧。": "No ratings yet. Go to \"Catalog\" to rate movies you like.",
    "暂无足够的上映年份数据，无法生成评分年代分布。": "Not enough release-year data to build the ratings-by-decade chart.",
    "更新密码": "Update Password",
    "最优 HitRate@10": "Best HitRate@10",
    "最优 MAE": "Best MAE",
    "最优 NDCG@10": "Best NDCG@10",
    "最优 Precision@10": "Best Precision@10",
    "最优 RMSE": "Best RMSE",
    "最优 Recall@10": "Best Recall@10",
    "最低评分": "Lowest Rating",
    "最偏好类型": "Favorite Genre",
    "最后评分日期（历史数据）": "Last Rating Date (historical)",
    "最常评分年代": "Top Rated Decade",
    "最常评分类型": "Most Rated Genre",
    "最新评分日期": "Latest Rating Date",
    "最活跃用户": "Most Active User",
    "最近评分日期": "Last Rating Date",
    "最近评分时间": "Last Rated At",
    "最高加权评分": "Top Weighted Score",
    "最高评分": "Highest Rating",
    "最高评分人次": "Top Rating Frequency",
    "最高评分数": "Most Ratings",
    "最高评分电影": "Top-Rated Movies",
    "月份": "Month",
    "月份：%{x}<br>评分数量：%{y:,}<extra></extra>": "Month: %{x}<br>Ratings: %{y:,}<extra></extra>",
    "有评分电影数": "Movies with Ratings",
    "未分类": "Uncategorized",
    "未找到匹配电影。": "No matching movies found.",
    "未找到匹配电影，请尝试其他关键词。": "No matching movies found. Try other keywords.",
    "未找到本地海报": "No local poster found",
    "未找到电影": "Movie ",
    "未找到符合条件的电影。": "No movies match the criteria.",
    "未知": "Unknown",
    "未设置": "Not set",
    "本地海报数量": "Local Posters",
    "本地海报目录：": "Local poster directory: ",
    "条评分": "ratings",
    "查看 IMDb 页面": "View IMDb Page",
    "查看推荐数据": "View Recommendation Data",
    "查看用户活跃度、评分行为与账号状态。": "View user activity, rating behavior, and account status.",
    "查看详情": "View Details",
    "查询、筛选并审查全站评分记录。": "Query, filter, and review all rating records.",
    "样本较少，仅供参考": "Small sample, for reference only",
    "根据为你生成的推荐结果统计的电影类型分布（取前 10 项）。": "Genre distribution of your recommendations (top 10).",
    "根据你的历史评分记录统计各电影类型的评分数量与平均评分，": "Genre statistics from your rating history — number of ratings and average rating per genre — ",
    "根据电影之间的评分相似度，为用户推荐与其历史喜欢的电影相似的其他电影。": "Recommends movies similar to the ones the user liked, based on rating similarity between movies.",
    "模型评估": "Model Evaluation",
    "模型评估指标摘要": "Model Evaluation Summary",
    "次数": "Count",
    "欢迎回来": "Welcome Back",
    "欢迎回来，以下是当前平台上最受欢迎的电影。完整的数据统计与图表请前往“{tab}”标签页查看。": "Welcome back! Here are the most popular movies on the platform. For full statistics and charts, visit the \"{tab}\" tab.",
    "正在生成个性化推荐...": "Generating personalized recommendations...",
    "正在编辑《": "Editing \"",
    "正常": "OK",
    "注册时间": "Registered",
    "活跃": "Active",
    "活跃用户数（评分数 ≥ 100）": "Active Users (≥ 100 ratings)",
    "活跃账号": "Active",
    "浏览、搜索并维护电影库中的元数据记录。": "Browse, search, and maintain metadata records in the movie catalog.",
    "浏览与搜索": "Browse & Search",
    "海报库已刷新。": "Poster library refreshed.",
    "海报文件": "Poster File",
    "海报文件名（位于“电影照片”目录下，留空则使用自动匹配）": "Poster filename (in the \"电影照片\" folder; leave empty for auto-matching)",
    "深度学习": "Deep Learning",
    "混合推荐": "Hybrid",
    "添加 / 编辑 / 删除电影记录": "Add / Edit / Delete Movie Records",
    "添加收藏": "Add Favorites",
    "点击电影卡片上的“查看详情”或“编辑”以显示详细信息。": "Click \"View Details\" or \"Edit\" on a movie card to show details.",
    "热门电影": "Popular Movies",
    "热门电影 ": "Popular Movies ",
    "热门电影排行": "Popular Movie Rankings",
    "热门电影类型分布": "Genre Distribution of Popular Movies",
    "爱情片爱好者": "Romance Fan",
    "生成推荐时发生错误，请尝试调整参数后重试。错误信息：": "Failed to generate recommendations. Please adjust the parameters and retry. Error: ",
    "用户": "User",
    "用户ID": "User ID",
    "用户ID必须为整数。": "User ID must be an integer.",
    "用户ID（精确匹配）": "User ID (exact match)",
    "用户偏好类型分布": "Your Preferred Genres",
    "用户偏好类型已更新。": "User preferences updated.",
    "用户名": "Username",
    "用户总数": "Total Users",
    "用户数": "Users",
    "用户数据管理": "User Data Management",
    "用户活跃度分布": "User Activity Distribution",
    "用户评分分布": "Your Rating Distribution",
    "用户评分记录": "User Rating Records",
    "用户详情 ": "User Details ",
    "电影": "Movie",
    "电影 ": "Movie ",
    "电影ID": "Movie ID",
    "电影ID或标题": "movie ID or title",
    "电影库": "Movie Catalog",
    "电影总数": "Total Movies",
    "电影按上映年代（10年为一组）的数量趋势": "Movie counts by release decade (10-year buckets)",
    "电影探索新手": "Movie Newbie",
    "电影推荐系统": "Movie Recommendation System",
    "电影数": "Movies",
    "电影数据管理": "Movie Data Management",
    "电影数量": "Movies",
    "电影标题": "Movie Title",
    "电影类型": "Genres",
    "电影类型分布": "Movie Genre Distribution",
    "电影详情 ": "Movie Details ",
    "电影详情页": "Movie Details",
    "电影（按ID、标题或类型）": " movies (by ID, title or genre)",
    "登录": "Sign In",
    "登录 FilmTrace，继续探索你的个性化电影推荐。": "Sign in to FilmTrace and continue exploring your personalized movie recommendations.",
    "相似度": "Similarity",
    "相似度 ": "Similarity ",
    "相似电影": "Similar Movies",
    "矩阵分解": "Matrix Factorization",
    "科幻爱好者": "Sci-Fi Fan",
    "移除收藏": "Remove",
    "稀疏度分析": "Sparsity Analysis",
    "筛选评分": "Filter by Rating",
    "算法配置": "Algorithm Config",
    "管理后台首页": "Admin Home",
    "类型": "Genre",
    "类型数量": "Genre Count",
    "类型评分分布": "Ratings by Genre",
    "类型：": "Genres: ",
    "类型：%{y}<br>平均评分：%{x:.2f}<br>已评分电影数：%{customdata[0]}<extra></extra>": "Genre: %{y}<br>Avg rating: %{x:.2f}<br>Rated movies: %{customdata[0]}<extra></extra>",
    "类型：%{y}<br>数量：%{x:,}<extra></extra>": "Genre: %{y}<br>Count: %{x:,}<extra></extra>",
    "系统会综合你的注册偏好（": "The system combines your registration preferences (",
    "系统将生成你的类型偏好分析。": "Your genre preference analysis will appear once you rate more movies.",
    "系统统计仪表盘": "System Statistics Dashboard",
    "经典电影探索者": "Classic Explorer",
    "结合协同过滤与电影元数据特征的混合推荐模型，缓解冷启动问题。": "A hybrid model combining collaborative filtering with movie metadata features to ease the cold-start problem.",
    "结合协同过滤与电影元数据特征（类型、用户/电影统计特征），用回归模型综合预测评分。": "Combines collaborative filtering with movie metadata (genres, user/movie stats) in a regression model to predict ratings.",
    "综合模型表现雷达图": "Overall Model Performance Radar",
    "编辑": "Edit",
    "缺失值数量": "Missing Values",
    "置信度 ": "Confidence ",
    "覆盖类型数": "Genres Covered",
    "访客": "Guest",
    "评价 / 评论（可选）": "Rating / Review (optional)",
    "评价已保存，可在“我的评分”页面查看。": "Rating saved. View it on the \"My Ratings\" page.",
    "评估结果明细": "Detailed Results",
    "评估评分矩阵稀疏度对协同过滤效果的影响。": "Assesses how rating-matrix sparsity affects collaborative filtering.",
    "评分": "Rating",
    "评分人数": "Ratings",
    "评分值": "Rating",
    "评分值：%{x}<br>评分数量：%{y:,}<extra></extra>": "Rating: %{x}<br>Count: %{y:,}<extra></extra>",
    "评分分布": "Rating Distribution",
    "评分分布、用户行为、电影类型分布等多维度可视化分析。": "Multi-dimensional visual analytics: rating distributions, user behavior, genre distributions, and more.",
    "评分年代分布": "Ratings by Decade",
    "评分总数": "Total Ratings",
    "评分数": "Ratings",
    "评分数量": "Ratings",
    "评分数量区间": "Rating Count Bucket",
    "评分数量较少的类型仅作为参考。": "Genres with few ratings are for reference only.",
    "评分最高电影 ": "Top-Rated Movies ",
    "评分次数": "Ratings",
    "评分行为分析": "Rating Behavior Analysis",
    "评分记录总数": "Total Ratings",
    "评分记录查询": "Rating Records Query",
    "评分记录管理": "Rating Records Management",
    "评分预测": "Rating Prediction",
    "该电影类型符合你注册时选择的偏好：": "Matches the genres you picked at registration: ",
    "该算法会根据你的历史评分数据计算预测评分，并按预测评分从高到低排序推荐。": "This algorithm computes predicted ratings from your rating history and recommends movies ranked from highest to lowest prediction.",
    "说明你的评分历史和偏好类型与该类电影具有较高匹配度。": "which matches your rating history and genre preferences well.",
    "请先登录后再收藏": "Please sign in to add favorites",
    "请先登录后再评价": "Please sign in to rate",
    "请解压 MovieLens 100K 数据集，确保 `data/raw/ml-100k/u.data` 存在，然后重启应用。": "Please extract the MovieLens 100K dataset, make sure `data/raw/ml-100k/u.data` exists, then restart the app.",
    "请输入管理员账号密码以打开管理后台。": "Enter the admin username and password to open the console. ",
    "还没有账号？请切换到注册页面创建账号。": "No account yet? Switch to the Sign Up tab to create one.",
    "选择你感兴趣的电影类型，帮助我们为你生成专属推荐。": "Pick the genres you like to help us tailor your recommendations.",
    "选择你的电影偏好，让系统为你生成更精准的推荐。": "Tell us your movie preferences so we can recommend more accurately.",
    "选择用户": "Select User",
    "通过矩阵分解学习每个用户和每部电影的隐含特征（隐因子），": "learns latent features (factors) of every user and movie through matrix factorization, ",
    "通过矩阵分解学习用户与电影的潜在因子，预测精度最高。": "Learns latent factors of users and movies via matrix factorization for the best prediction accuracy.",
    "邮箱": "Email",
    "部": "movies",
    "部电影": "movies",
    "除非在 .env 文件中通过 ADMIN_USERNAME / ADMIN_PASSWORD 覆盖，": "Unless overridden via ADMIN_USERNAME / ADMIN_PASSWORD in the .env file, ",
    "预测误差对比": "Prediction Error Comparison",
    "首次生成可能需要数秒，后续相同参数将直接读取缓存。": "The first run may take a few seconds; identical parameters will hit the cache afterwards.",
    "高偏好": "High",
    "高分电影数": "High-Rated Movies",
    "高分筛选型用户": "High-Bar Rater",
    "高分类型数量": "High-Rated Genres",
    "高活跃": "Very Active",
    "高级参数设置": "Advanced Settings",
    "默认演示账号为 admin / admin123。": "the default demo account is admin / admin123.",
    "默认算法": "Default Algorithm",
    "（基于物品-物品相似度），并按相似度与预测评分排序后推荐给你。": "(item-item similarity), and recommends them ranked by similarity and predicted rating.",
    "（按标题或电影ID）": " (by title or movie ID)",
    "（管理员设置）": " (set by admin)",
    "（结果已缓存，刷新页面不会重新训练）。": "(results are cached; refreshing the page will not retrain).",
    "）": "), ",
    "）、": "), ",
    "）。": ").",
    "，显示最相似的 ": ", showing the ",
    "：": ": ",
    "🔄 刷新海报库": "🔄 Refresh Poster Library",
}


_TABLES: dict[str, dict[str, str]] = {"zh": _ZH, "en": {**_EN, **_EN_LITERALS}}


def t(key: str) -> str:
    """Look up a UI string in the current language.

    Falls back to the Chinese table, then to the key itself (which keeps the
    Chinese-as-key convention working for literals not yet translated).
    """
    table = _TABLES.get(_CURRENT_LANG, _ZH)
    if key in table:
        return table[key]
    return _ZH.get(key, key)


class _TextProxy:
    """Dict-like live view over the current language's string table.

    Existing code uses ``T["key"]`` / ``T.get("key")`` extensively; this proxy
    keeps that syntax working while resolving against the active language on
    every access (so a language switch takes effect on the next rerun without
    touching call sites).
    """

    def __getitem__(self, key: str) -> str:
        return t(key)

    def get(self, key: str, default: str | None = None) -> str | None:
        value = t(key)
        if value == key and key not in _ZH:
            return default
        return value

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in _ZH or key in _TABLES.get(_CURRENT_LANG, _ZH))


T = _TextProxy()


# Display-only column header translations applied to DataFrames right before
# they are rendered (st.dataframe / QTableWidget). Internal column names used
# in computations stay in English so the data-processing code is unaffected,
# which means no translation is needed when the UI language is English.
COLUMN_LABELS: dict[str, str] = {
    "Movie ID": "电影ID",
    "Title": "标题",
    "Movie Title": "电影标题",
    "Genres": "类型",
    "Genre": "类型",
    "Average Rating": "平均评分",
    "Avg Rating": "平均评分",
    "Ratings": "评分数",
    "Rated Movies": "已评分电影数",
    "Release Year": "上映年份",
    "Similarity": "相似度",
    "Rank": "排名",
    "Recommendation Score": "推荐评分",
    "Raw Model Score": "原始模型评分",
    "Confidence": "置信度",
    "Reason": "推荐理由",
    "User Rating": "用户评分",
    "Rated At": "评分时间",
    "user_id": "用户ID",
    "Activity Tier": "活跃等级",
    "Last Activity": "最近活动",
    "Algorithm": "算法",
    "Evaluated Ratings": "评估样本数",
    "Timestamp": "时间",
    "Operation": "操作",
    "Administrator": "管理员",
    "Target": "目标",
    "Comparison": "对比",
    "MSE Difference": "MSE 差值",
    "t-stat": "t 统计量",
    "p-value": "p 值",
    "Effect Size": "效应量",
    "Factors": "因子数",
    "Precision@10": "Precision@10",
    "Recall@10": "Recall@10",
    "HitRate@10": "HitRate@10",
    "NDCG@10": "NDCG@10",
    "Best Algorithm": "最优算法",
    "Top-N Coverage": "Top-N 覆盖率",
    "Avg Diversity": "平均多样性",
    "Neighbor Count": "近邻数",
    "Average Similarity": "平均相似度",
    "Preference Score": "偏好得分",
    "Date": "日期",
    "Status": "状态",
}


def translate_columns(df):
    """Return a copy of `df` with column headers translated to Chinese.

    Only applies when the UI language is Chinese — internal column names are
    already English, so under English the frame is returned unchanged.
    """
    if _CURRENT_LANG != "zh":
        return df
    rename_map = {col: COLUMN_LABELS[col] for col in df.columns if col in COLUMN_LABELS}
    return df.rename(columns=rename_map) if rename_map else df


# Manually curated Chinese titles for a subset of well-known MovieLens 100K
# movies. Matching is done by substring against the original English title
# (which includes the release year, e.g. "Toy Story (1995)"), so each entry
# only needs the distinctive part of the title.
MOVIE_TITLE_ZH: dict[str, str] = {
    "Toy Story": "玩具总动员",
    "Star Wars": "星球大战",
    "Schindler's List": "辛德勒的名单",
    "Shawshank Redemption": "肖申克的救赎",
    "Casablanca": "卡萨布兰卡",
    "Usual Suspects": "非常嫌疑犯",
    "Godfather": "教父",
    "Rear Window": "后窗",
    "Raiders of the Lost Ark": "夺宝奇兵",
    "Silence of the Lambs": "沉默的羔羊",
    "Fargo": "冰血暴",
    "Return of the Jedi": "绝地归来",
    "Contact": "超时空接触",
    "Titanic": "泰坦尼克号",
    "Good Will Hunting": "心灵捕手",
    "L.A. Confidential": "洛城机密",
    "12 Angry Men": "十二怒汉",
    "Vertigo": "迷魂记",
    "Amadeus": "莫扎特传",
    "Pulp Fiction": "低俗小说",
}


# English MovieLens genre names -> Chinese display names.
GENRE_ZH: dict[str, str] = {
    "Action": "动作",
    "Adventure": "冒险",
    "Animation": "动画",
    "Children's": "儿童",
    "Comedy": "喜剧",
    "Crime": "犯罪",
    "Documentary": "纪录片",
    "Drama": "剧情",
    "Fantasy": "奇幻",
    "Film-Noir": "黑色电影",
    "Horror": "恐怖",
    "Musical": "音乐",
    "Mystery": "悬疑",
    "Romance": "爱情",
    "Sci-Fi": "科幻",
    "Thriller": "惊悚",
    "War": "战争",
    "Western": "西部",
    "Unknown": "未分类",
    "unknown": "未分类",
    "未知类型": "未分类",
}


def get_display_title(title: str) -> str:
    """Return the display title for a movie in the current language.

    Under Chinese, well-known movies get their curated Chinese title; under
    English (and for unmapped movies) the original title is returned.
    """
    if _CURRENT_LANG != "zh":
        return title
    for fragment, zh_title in MOVIE_TITLE_ZH.items():
        if fragment in title:
            return zh_title
    return title


# Chinese genre options offered on the registration / onboarding page.
# Each entry must be a key in GENRE_EN_BY_ZH so it can be mapped back to a
# MovieLens genre column for cold-start filtering.
ONBOARDING_GENRES: list[str] = [
    "动作",
    "冒险",
    "动画",
    "儿童",
    "喜剧",
    "犯罪",
    "纪录片",
    "剧情",
    "奇幻",
    "黑色电影",
    "恐怖",
    "音乐",
    "悬疑",
    "爱情",
    "科幻",
    "惊悚",
    "战争",
    "西部",
]


# Reverse lookup: Chinese genre name -> English MovieLens genre column name.
GENRE_EN_BY_ZH: dict[str, str] = {
    zh: en for en, zh in GENRE_ZH.items() if en not in ("unknown", "Unknown", "未知类型")
}

# English genre name -> canonical Chinese genre name (for storing preferences
# gathered while the UI is in English).
_GENRE_ZH_BY_EN: dict[str, str] = {en: zh for zh, en in GENRE_EN_BY_ZH.items()}


def display_genre(genre_en: str) -> str:
    """Return the display name of one English genre in the current language."""
    if _CURRENT_LANG == "zh":
        return GENRE_ZH.get(genre_en, genre_en)
    return genre_en


def onboarding_genre_options() -> list[str]:
    """Genre options for registration/onboarding in the current language."""
    if _CURRENT_LANG == "zh":
        return list(ONBOARDING_GENRES)
    return [GENRE_EN_BY_ZH[zh] for zh in ONBOARDING_GENRES]


def canonical_genre_zh(display_name: str) -> str:
    """Map a displayed genre name (Chinese or English) back to canonical Chinese."""
    if display_name in GENRE_EN_BY_ZH:
        return display_name
    return _GENRE_ZH_BY_EN.get(display_name, display_name)


def translate_genres(genres: str) -> str:
    """Translate a comma-separated English genre string for display.

    Under Chinese, genres are translated and joined by '，'; under English the
    names are returned as-is, joined by ', '.
    """
    if not genres:
        return "未知" if _CURRENT_LANG == "zh" else "Unknown"
    parts = [part.strip() for part in genres.split(",") if part.strip()]
    if not parts:
        return "未知" if _CURRENT_LANG == "zh" else "Unknown"
    if _CURRENT_LANG == "zh":
        return "，".join(GENRE_ZH.get(part, part) for part in parts)
    return ", ".join(parts)
