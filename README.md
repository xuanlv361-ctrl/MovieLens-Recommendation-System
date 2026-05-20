# MovieLens 100K Recommendation System

University data-mining course project: collaborative filtering and optional matrix factorization on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset.

## Features

- **User-based collaborative filtering** (cosine / Pearson similarity, k-nearest neighbors)
- **Item-based collaborative filtering**
- **Optional SVD** matrix factorization (`sklearn.decomposition.TruncatedSVD`)
- **Evaluation**: RMSE and MAE on a held-out test set
- **Visualizations**: EDA and model comparison plots saved to `figures/`
- **Reproducibility**: `random_state=42` for splits and SVD

## Project structure

```text
movielens_project/
├── data/
│   ├── raw/              # Place extracted ml-100k here
│   └── processed/        # Optional cached splits
├── figures/              # Generated plots
├── app.py                # Streamlit similar-movie demo
├── notebooks/
│   └── 01_movielens_recommendation.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── similarity.py
│   ├── metrics.py
│   ├── user_based_cf.py
│   ├── item_based_cf.py
│   ├── svd_model.py
│   └── visualization.py
├── README.md
├── requirements.txt
└── environment.yml
```

## Dataset setup

1. Download **ml-100k** from [GroupLens](https://grouplens.org/datasets/movielens/100k/).
2. Extract the archive so that `u.data` is located at:

   ```text
   data/raw/ml-100k/u.data
   ```

3. Required files for the notebook: `u.data`, `u.item`, `u.genre`.

The raw dataset is **not** included in this repository (see `.gitignore`).

## Environment setup

### Option A — Conda

```bash
conda env create -f environment.yml
conda activate movielens-dm
```

### Option B — pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

All dependencies install cleanly on Windows with Anaconda (no C extensions beyond standard scientific stack).

## Running the project

From the project root:

```bash
jupyter lab notebooks/01_movielens_recommendation.ipynb
```

Or:

```bash
jupyter notebook notebooks/01_movielens_recommendation.ipynb
```

Run all cells top to bottom. Figures are written to `figures/`.

**Runtime:** User- and item-based CF build full similarity matrices on the training set; expect several minutes on a typical laptop. Reduce `K_NEIGHBORS` or use a smaller test sample only for debugging—not for final reported metrics.

### Streamlit demo

Interactive **similar-movie** recommendations (item–item collaborative filtering):

```bash
streamlit run app.py
```

From the project root, with the dataset in `data/raw/ml-100k/`. The first launch builds the item similarity matrix (cached for later runs). Select a title in the sidebar to see the top 10 similar movies.

## Hyperparameters

Configured in `src/config.py`:

| Parameter | Default |
|-----------|---------|
| `RANDOM_STATE` | 42 |
| `TEST_SIZE` | 0.2 |
| `K_NEIGHBORS` | 20 |
| `MIN_COMMON_RATINGS` | 5 |
| `SVD_N_COMPONENTS` | 50 |

## Metrics

- **RMSE** — root mean squared error between predicted and actual ratings
- **MAE** — mean absolute error

Lower values indicate better rating prediction on the hold-out set.

## Citation

If you use this dataset in coursework or publications, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

## References

- Sarwar, B. et al. Item-based collaborative filtering recommendation algorithms. WWW 2001.
- Koren, Y. et al. Matrix factorization techniques for recommender systems. Computer 2009.

## License

Project code: educational use. MovieLens data: see [GroupLens terms](https://grouplens.org/datasets/movielens/).
