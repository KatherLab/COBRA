# Multilabel Classification Support

The COBRA framework now supports both single-label (multiclass) and multilabel classification. This document explains how to configure and use these modes.

## Configuration

### Single-Label Classification (Original Behavior)
For traditional multiclass classification, use the `target_column` parameter:

```yaml
train:
  csv_path: "/path/to/metadata.csv"
  target_column: "TUMOR_TYPE"  # Single column with class labels
  patient_id_column: "PATIENT_ID"
  # ... other parameters
```

### Multi-Label Classification (New Feature)
For multilabel classification, use the `target_columns` parameter with a list of binary target columns:

```yaml
train:
  csv_path: "/path/to/metadata.csv"
  target_columns: ["MALIGNANT", "AGGRESSIVE", "METASTATIC"]  # Multiple binary labels
  patient_id_column: "PATIENT_ID"
  # ... other parameters
```

## Data Format Requirements

### Single-Label Data
- CSV should have one column with categorical labels
- Labels will be automatically encoded using LabelEncoder
- Example: `TUMOR_TYPE` column with values like "Adenocarcinoma", "Squamous", etc.

### Multi-Label Data
- CSV should have multiple columns, each representing a binary label (0/1)
- Each row represents a patient with multiple potential labels
- Missing values in target columns will be dropped
- Example:
  ```
  PATIENT_ID, MALIGNANT, AGGRESSIVE, METASTATIC
  P001,      1,         0,          1
  P002,      0,         1,          0
  P003,      1,         1,          1
  ```

## Model Architecture Changes

### Single-Label Mode
- Uses CrossEntropyLoss
- Output layer size = number of classes
- Final activation: Softmax (implicit in CrossEntropyLoss)
- Metrics: Multiclass accuracy and AUROC

### Multi-Label Mode
- Uses BCEWithLogitsLoss (Binary Cross Entropy with Logits)
- Output layer size = number of labels
- Final activation: Sigmoid (applied during inference)
- Metrics: Multilabel accuracy and AUROC

## Cross-Validation Strategy

### Single-Label
- Uses StratifiedKFold to ensure balanced class distribution across folds

### Multi-Label
- Uses regular KFold since stratified multilabel cross-validation is complex
- Ensures reproducible splits with fixed random seeds

## Output Files

### Single-Label Results
- `fold_X_test_results.csv`: Contains patient_id, ground_truth, prediction, loss
- `fold_aurocs.csv`: AUROC scores for each fold
- Labels are in original format (inverse transformed)

### Multi-Label Results
- `fold_X_test_results.csv`: Contains patient_id, loss, and for each label:
  - `ground_truth_LABEL_NAME`: Binary ground truth (0/1)
  - `prediction_LABEL_NAME`: Binary prediction (0/1, threshold=0.5)
  - `probability_LABEL_NAME`: Predicted probability [0,1]
- `fold_X_individual_aurocs.csv`: Individual AUROC scores for each target in each fold
- `all_folds_individual_aurocs.csv`: Combined individual AUROC scores across all folds
- `fold_aurocs.csv`: Macro-averaged AUROC scores for each fold

### Deployment Results (Additional)
For multilabel deployment, additional files are generated:
- `fold_X_individual_metrics.csv`: Individual AUROC and F1 scores for each target in each fold
- `all_folds_individual_metrics.csv`: Combined individual metrics across all folds
- `auroc_scores.csv`: Includes both AUROC and F1 macro-averaged scores

### Label Encoders
- Single-label: Saves single LabelEncoder as `label_encoders.pkl`
- Multi-label: Saves dictionary of encoders (if needed) as `label_encoders.pkl`

## Usage Examples

### Training Single-Label Model
```bash
python -m cobra.crossval.train -c configs/example.yml
```

### Training Multi-Label Model
```bash
python -m cobra.crossval.train -c configs/example_multilabel.yml
```

### Deploying Models
```bash
python -m cobra.crossval.deploy -c configs/example_multilabel.yml
```

## Individual Target Metrics

### Training Phase
For multilabel classification, the training script now reports:
- **Macro-averaged AUROC**: Overall performance across all labels
- **Individual AUROC per target**: Performance for each specific label
- **Console output**: Detailed per-fold, per-target AUROC scores
- **Summary statistics**: Average AUROC per target across all folds

### Deployment Phase
For multilabel classification, the deployment script additionally reports:
- **F1 scores**: Both macro-averaged and per-target F1 scores
- **Individual metrics**: AUROC and F1 for each target separately
- **Detailed reporting**: Console output with per-fold, per-target metrics
- **Comprehensive summaries**: Average performance per target across folds

### Console Output Example
```
Fold 0 Test AUROC (macro): 0.8234
Fold 0 Test AUROC (MALIGNANT): 0.8456
Fold 0 Test AUROC (AGGRESSIVE): 0.7892
Fold 0 Test AUROC (METASTATIC): 0.8354

Average AUROC per target across all folds:
  MALIGNANT: 0.8234
  AGGRESSIVE: 0.7845
  METASTATIC: 0.8123
  Macro average: 0.8067
```

## Key Implementation Details

1. **Reproducibility**: All random operations use fixed seeds (42) for consistent results
2. **GPU Support**: Automatically uses GPU if available
3. **Memory Efficiency**: Uses inference mode during testing
4. **Threshold**: Multilabel predictions use 0.5 threshold for binary classification
5. **AUROC Calculation**: Uses macro-averaged AUROC for multilabel classification
6. **Individual Metrics**: Detailed per-target performance tracking and reporting
7. **Error Handling**: Graceful handling of cases with only one class present

## Backwards Compatibility

All existing single-label configurations will continue to work without modification. The system automatically detects whether to use single-label or multilabel mode based on the presence of `target_column` vs `target_columns` in the configuration.