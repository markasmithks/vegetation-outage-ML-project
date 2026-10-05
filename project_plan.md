01_validate_historical_analysis.ipynb
        ↓
Historical baseline / reproduce clustering approach

02_define_ml_problem.ipynb
        ↓
Define prediction unit, target, leakage rules,
train/validation/test strategy

03_feature_engineering.ipynb
        ↓
Create model-ready dataset

04_model_development.ipynb
        ↓
Baseline → candidate models → evaluation

05_final_model_evaluation.ipynb
        ↓
Untouched test set + explainability/error analysis

        ↓
Move mature code OUT of notebooks

src/
    data/
    features/
    models/
    evaluation/

        ↓
Automated training pipeline
        ↓
Azure ML experiment/training
        ↓
Model registry
        ↓
Managed endpoint/API
        ↓
Monitoring/logging
        ↓
GitHub + documentation

