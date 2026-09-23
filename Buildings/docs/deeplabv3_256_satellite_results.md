# DeepLabV3 ResNet50 - Satellite - Tile 256 Results

## Experiment Setup

- Task: Building extraction
- Model: DeepLabV3
- Backbone: ResNet50
- Source type: Satellite
- Tile size: 256
- Dataset type: Classified Tiles
- Training normalization: image / 255.0 only
- Pretrained: False
- Auxiliary loss: False
- Batch size: 8

## Results

| Stage | Threshold | IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Standard Model | 0.45 | 51.63% | 76.39% | 61.44% | 68.10% |
| Standard Model + Optuna | 0.35 | 59.16% | 82.27% | 67.81% | 74.34% |
| Standard Model + Optuna + Post-processing | 0.35 | 59.47% | 83.64% | 67.31% | 74.59% |

## Best Final Result

The best final result was obtained using the Optuna-optimized DeepLabV3 ResNet50 model followed by post-processing.

- Best threshold: 0.35
- Best post-processing method: area_75_hole_5
- Final IoU: 59.47%
- Final Precision: 83.64%
- Final Recall: 67.31%
- Final F1-score: 74.59%

## Final Outputs

- Best model:
  Buildings/outputs/optuna/deeplabv3_256_satellite_optuna/best_model.pth

- Raw prediction:
  Buildings/outputs/optuna/deeplabv3_256_satellite_optuna/testing/optuna_testing.gdb/optuna_final_prediction_raw_thr_0_35

- Final post-processed prediction:
  Buildings/outputs/optuna/deeplabv3_256_satellite_optuna/testing/optuna_testing.gdb/optuna_final_prediction_postprocessed

- Final summary CSV:
  Buildings/outputs/optuna/deeplabv3_256_satellite_optuna/testing/optuna_final_testing_summary.csv

- Final summary JSON:
  Buildings/outputs/optuna/deeplabv3_256_satellite_optuna/testing/optuna_final_testing_summary.json