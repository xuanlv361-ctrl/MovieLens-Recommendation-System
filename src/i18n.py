"""Shared Simplified Chinese UI strings for the Streamlit and PyQt5 demo apps.

Centralizing the common navigation labels, buttons, and status messages here
keeps the two front ends consistent and makes future wording changes a
one-line edit instead of a find-and-replace across both apps.

Algorithm/metric names (UserCF, ItemCF, SVD, NeuralCF, Hybrid, RMSE, MAE,
NDCG, ...) are intentionally kept in English/abbreviated form, matching
common usage in Chinese technical writing.
"""

from __future__ import annotations

T: dict[str, str] = {
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
}


def t(key: str) -> str:
    """Look up a shared UI string, falling back to the key itself."""
    return T.get(key, key)


# Display-only column header translations applied to DataFrames right before
# they are rendered (st.dataframe / QTableWidget). Internal column names used
# in computations stay in English so the data-processing code is unaffected.
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
    """Return a copy of `df` with any recognized column headers translated to Chinese."""
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
    "Unknown": "未知",
    "unknown": "未知",
    "未知类型": "未知",
}


def get_display_title(title: str) -> str:
    """Return a Chinese title for well-known movies, otherwise the original title."""
    for fragment, zh_title in MOVIE_TITLE_ZH.items():
        if fragment in title:
            return zh_title
    return title


# Reverse lookup: Chinese genre name -> English MovieLens genre column name.
GENRE_EN_BY_ZH: dict[str, str] = {
    zh: en for en, zh in GENRE_ZH.items() if en not in ("unknown", "Unknown", "未知类型")
}


def translate_genres(genres: str) -> str:
    """Translate a comma-separated English genre string to Chinese, joined by '，'."""
    if not genres:
        return "未知"
    parts = [part.strip() for part in genres.split(",") if part.strip()]
    if not parts:
        return "未知"
    return "，".join(GENRE_ZH.get(part, part) for part in parts)
