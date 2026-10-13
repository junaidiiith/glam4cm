# Independent Uncorrected Significance Results

These tables use independent raw Wilcoxon p-values with `p < 0.05` and do not apply FDR correction.
Use these as exploratory per-comparison results; the FDR-corrected report remains the stricter family-wise interpretation.

## k-hop Effects

| model_type   |   k_to |   num_matched_tests |   mean_delta |   median_delta |   pvalue_uncorrected | significant_uncorrected_0.05   |
|:-------------|-------:|--------------------:|-------------:|---------------:|---------------------:|:-------------------------------|
| GNN(FTLM)    |      1 |                  23 |       0.0522 |         0.0107 |               0.0031 | True                           |
| GNN(FTLM)    |      2 |                  23 |       0.0397 |         0.0055 |               0.2345 | False                          |
| GNN(FTLM)    |      3 |                  23 |       0.0083 |        -0.0077 |               0.6010 | False                          |
| LLM          |      1 |                  30 |       0.0346 |         0.0235 |               0.0000 | True                           |
| LLM          |      2 |                  30 |       0.0263 |         0.0236 |               0.0234 | True                           |
| LLM          |      3 |                  30 |       0.0345 |         0.0247 |               0.0002 | True                           |

## Individual Semantic Parameter Effects

| model_type   | parameter_added   |   num_matched_tests |   mean_delta |   median_delta |   pvalue_uncorrected | significant_uncorrected_0.05   |
|:-------------|:------------------|--------------------:|-------------:|---------------:|---------------------:|:-------------------------------|
| GNN(FTLM)    | EL                |                  18 |       0.0059 |         0.0038 |               0.5226 | False                          |
| GNN(FTLM)    | ET                |                  21 |       0.0175 |        -0.0006 |               1.0000 | False                          |
| GNN(FTLM)    | NA                |                  28 |       0.0214 |         0.0118 |               0.0103 | True                           |
| GNN(FTLM)    | NT                |                  36 |       0.0124 |         0.0054 |               0.1189 | False                          |
| LLM          | EL                |                  18 |      -0.0124 |        -0.0015 |               0.5509 | False                          |
| LLM          | ET                |                  21 |      -0.0000 |         0.0064 |               0.6333 | False                          |
| LLM          | NA                |                  36 |      -0.0021 |         0.0033 |               0.4051 | False                          |
| LLM          | NT                |                  47 |       0.0027 |         0.0015 |               0.3415 | False                          |

## Canonical Combination Effects Compared With NL

| model_type   | config_to      |   num_matched_tests |   mean_delta |   median_delta |   pvalue_uncorrected | significant_uncorrected_0.05   |
|:-------------|:---------------|--------------------:|-------------:|---------------:|---------------------:|:-------------------------------|
| GNN(FTLM)    | NL+NA          |                  28 |       0.0214 |         0.0118 |               0.0103 | True                           |
| GNN(FTLM)    | NL+NA+NT       |                  25 |       0.0293 |         0.0141 |               0.0125 | True                           |
| GNN(FTLM)    | NL+NA+NT+EL    |                  18 |       0.0269 |         0.0107 |               0.0066 | True                           |
| GNN(FTLM)    | NL+NA+NT+EL+ET |                  12 |       0.0187 |        -0.0019 |               1.0000 | False                          |
| LLM          | NL+NA          |                  36 |      -0.0021 |         0.0033 |               0.4051 | False                          |
| LLM          | NL+NA+NT       |                  33 |      -0.0020 |         0.0057 |               0.6720 | False                          |
| LLM          | NL+NA+NT+EL    |                  18 |      -0.0211 |        -0.0003 |               0.3465 | False                          |
| LLM          | NL+NA+NT+EL+ET |                  12 |      -0.0417 |        -0.0082 |               0.1763 | False                          |

## Omnibus Configuration Tests

| model_type   |   num_tests |   significant_uncorrected_0_05 |   median_raw_pvalue |
|:-------------|------------:|-------------------------------:|--------------------:|
| GNN(FTLM)    |          34 |                             30 |           8.225e-05 |
| LLM          |          42 |                             36 |           0.0005005 |

## GNN(FTLM) vs LLM

| task                 | dataset    | scope           |    n |   mean_llm |   mean_gnn_ftlm |   delta |   pvalue_uncorrected | significant_uncorrected_0.05   |
|:---------------------|:-----------|:----------------|-----:|-----------:|----------------:|--------:|---------------------:|:-------------------------------|
| Edge classification  | EAModelSet | task_dataset    |  110 |     0.8200 |          0.8460 |  0.0260 |               0.0014 | True                           |
| Edge classification  | Ecore-555  | task_dataset    |  180 |     0.9076 |          0.9380 |  0.0303 |               0.0000 | True                           |
| Edge classification  | ModelSet   | task_dataset    |  180 |     0.9292 |          0.9606 |  0.0314 |               0.0000 | True                           |
| Graph classification | Ecore-555  | task_dataset    |  180 |     0.9272 |          0.9525 |  0.0253 |               0.0000 | True                           |
| Graph classification | ModelSet   | task_dataset    |  180 |     0.7865 |          0.7795 | -0.0070 |               0.0001 | True                           |
| Node classification  | EAModelSet | task_dataset    |  200 |     0.7685 |          0.7707 |  0.0023 |               0.0216 | True                           |
| Node classification  | Ecore-555  | task_dataset    |  140 |     0.8933 |          0.9204 |  0.0271 |               0.0000 | True                           |
| Node classification  | ModelSet   | task_dataset    |  140 |     0.8832 |          0.8829 | -0.0003 |               0.3869 | False                          |
| Node classification  | OntoUML    | task_dataset    |  110 |     0.6878 |          0.7123 |  0.0244 |               0.0000 | True                           |
| Edge classification  | Overall    | task_overall    |  470 |     0.8954 |          0.9251 |  0.0297 |               0.0000 | True                           |
| Graph classification | Overall    | task_overall    |  360 |     0.8569 |          0.8660 |  0.0091 |               0.0027 | True                           |
| Node classification  | Overall    | task_overall    |  590 |     0.8103 |          0.8219 |  0.0117 |               0.0000 | True                           |
| Overall              | EAModelSet | dataset_overall |  310 |     0.7868 |          0.7974 |  0.0107 |               0.0000 | True                           |
| Overall              | Ecore-555  | dataset_overall |  500 |     0.9107 |          0.9383 |  0.0276 |               0.0000 | True                           |
| Overall              | ModelSet   | dataset_overall |  500 |     0.8650 |          0.8737 |  0.0087 |               0.0000 | True                           |
| Overall              | OntoUML    | dataset_overall |  110 |     0.6878 |          0.7123 |  0.0244 |               0.0000 | True                           |
| Overall              | Overall    | overall         | 1420 |     0.8503 |          0.8673 |  0.0170 |               0.0000 | True                           |
