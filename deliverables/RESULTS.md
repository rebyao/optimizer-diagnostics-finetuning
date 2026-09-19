# Results tables

Accuracy is a fraction (0–1). Diagnostics are from a separate 32-step run.

## Full-run performance

| optimizer | final_epoch | final_train_loss | final_validation_loss | final_validation_accuracy | best_epoch | best_validation_loss | best_validation_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adamw | 3 | 0.0769633 | 0.262449 | 0.908257 | 3 | 0.262449 | 0.908257 |
| muon | 3 | 0.136087 | 0.39393 | 0.844037 | 1 | 0.369564 | 0.870413 |

## Recorded epochs

| optimizer | epoch | train_loss | validation_loss | validation_accuracy |
| --- | --- | --- | --- | --- |
| adamw | 1 | 0.22044 | 0.25766 | 0.888761 |
| adamw | 2 | 0.114025 | 0.287294 | 0.895642 |
| adamw | 3 | 0.0769633 | 0.262449 | 0.908257 |
| muon | 1 | 0.254595 | 0.369564 | 0.870413 |
| muon | 2 | 0.16923 | 0.570577 | 0.838303 |
| muon | 3 | 0.136087 | 0.39393 | 0.844037 |

## 32-step arithmetic means by group

| optimizer | group | training_loss | gradient_norm | update_norm | relative_update_norm |
| --- | --- | --- | --- | --- | --- |
| adamw | all | 0.671843 | 1.40018 | 0.055392 | 0.000132995 |
| adamw | fallback | 0.671843 | 0.835177 | 0.00785941 | 2.57765e-05 |
| adamw | fallback_decay | 0.671843 | 0.810024 | 0.00755698 | 2.57237e-05 |
| adamw | fallback_no_decay | 0.671843 | 0.201484 | 0.00210702 | 2.58106e-05 |
| adamw | matrix | 0.671843 | 1.1165 | 0.054812 | 0.000193184 |
| muon | all | 0.651614 | 1.53027 | 0.493242 | 0.00118445 |
| muon | fallback | 0.651614 | 0.91154 | 0.00791194 | 2.59488e-05 |
| muon | fallback_decay | 0.651614 | 0.886176 | 0.00761151 | 2.59093e-05 |
| muon | fallback_no_decay | 0.651614 | 0.209879 | 0.00210483 | 2.57839e-05 |
| muon | matrix | 0.651614 | 1.22056 | 0.493177 | 0.00173879 |

## Checkpoint sharpness

| optimizer | epsilon | baseline_loss | mean_symmetric_delta | std_symmetric_delta | max_sampled_delta |
| --- | --- | --- | --- | --- | --- |
| adamw | 0 | 0.225329 | 0 | 0 | 0 |
| adamw | 0.001 | 0.225329 | -5.2125e-08 | 1.50691e-06 | 0.000303127 |
| adamw | 0.003 | 0.225329 | -6.97299e-07 | 1.16862e-05 | 0.000902378 |
| adamw | 0.01 | 0.225329 | 8.04843e-06 | 0.000130616 | 0.00297951 |
| adamw | 0.03 | 0.225329 | 0.000121 | 0.00111749 | 0.00987973 |
| adamw | 0.1 | 0.225329 | 0.00741035 | 0.00742546 | 0.0391211 |
| muon | 0 | 0.319872 | 0 | 0 | 0 |
| muon | 0.001 | 0.319872 | -3.3144e-06 | 3.39669e-06 | 0.000207772 |
| muon | 0.003 | 0.319872 | -2.66179e-05 | 3.06718e-05 | 0.000616739 |
| muon | 0.01 | 0.319872 | -0.000111017 | 0.000141172 | 0.00196704 |
| muon | 0.03 | 0.319872 | 0.000121225 | 0.000772037 | 0.00464802 |
| muon | 0.1 | 0.319872 | 0.00435421 | 0.0059285 | 0.0243333 |

## Paired directional uncertainty

| epsilon | muon_minus_adamw_mean | ci95_low | ci95_high |
| --- | --- | --- | --- |
| 0.001 | -3.26228e-06 | -5.83191e-06 | -6.92644e-07 |
| 0.003 | -2.59206e-05 | -4.78743e-05 | -3.96689e-06 |
| 0.01 | -0.000119066 | -0.000228815 | -9.31682e-06 |
| 0.03 | 2.25176e-07 | -0.000966222 | 0.000966673 |
| 0.1 | -0.00305614 | -0.00976395 | 0.00365167 |

