msmeqe/
├── __init__.py
├── utils/
│   ├── file_utils.py
│   ├── logging_utils.py
│   └── stats_utils.py            # (optional: DF/CF cache helpers)
├── retrieval/
│   ├── create_index.py
│   ├── bm25_scorer.py
│   ├── evaluator.py
│   └── metrics.py
├── expansion/
│   ├── rm_expansion.py
│   ├── kb_expansion.py
│   ├── embedding_candidates.py
│   ├── knapsack.py
│   └── msmeqe_expansion.py       # core MS-MEQE expansion model
├── features/
│   ├── feature_extraction.py
│   └── create_training_data_msmeqe.py
├── models/
│   ├── train_value_weight_models.py
│   ├── train_budget_model.py
│   └── load_models.py
├── reranking/
│   ├── semantic_encoder.py
│   └── bi_encoder_reranker.py
└── pipeline/
    ├── ms_meqe_pipeline.py
    └── baselines.py              # BM25, BM25+RM3 only
