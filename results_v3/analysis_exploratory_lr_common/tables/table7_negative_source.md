### table7_negative_source: levels, common lr vs v3 (mean over runs)

| slot | regime | regime_label | auroc_mean_common | specificity_mean_common | balanced_accuracy_mean_common | auroc_mean_v3 | specificity_mean_v3 | balanced_accuracy_mean_v3 | auroc_diff | specificity_diff | balanced_accuracy_diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| isolated | sup_imnet | SUP-imnet | 0.5280 | 0.2922 | 0.5236 | 0.6886 | 0.1912 | 0.5616 | -0.1606 | 0.1010 | -0.0380 |
| isolated | ssl_imnet | SSL-imnet | 0.5344 | 0.2361 | 0.5137 | 0.6958 | 0.2222 | 0.5705 | -0.1614 | 0.0138 | -0.0568 |
| isolated | ssl_colon | SSL-colon | 0.6081 | 0.3057 | 0.5637 | 0.6187 | 0.1400 | 0.5304 | -0.0106 | 0.1656 | 0.0333 |
| sequence | sup_imnet | SUP-imnet | 0.7158 | 0.5711 | 0.6630 | 0.8828 | 0.6511 | 0.7915 | -0.1670 | -0.0799 | -0.1285 |
| sequence | ssl_imnet | SSL-imnet | 0.7113 | 0.5353 | 0.6633 | 0.8258 | 0.5180 | 0.7184 | -0.1145 | 0.0173 | -0.0551 |
| sequence | ssl_colon | SSL-colon | 0.7410 | 0.5924 | 0.7070 | 0.7754 | 0.5177 | 0.7192 | -0.0344 | 0.0747 | -0.0122 |
