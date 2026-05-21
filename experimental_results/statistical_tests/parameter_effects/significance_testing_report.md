# Significance Testing Report: Semantic Parameters and k-hop Effects

## Data and Design

- Input rows: 3330 total from node, edge, link, and graph classification CSV files.
- Response variable: F1.
- Unit of pairing: same task, dataset, target, seed, and the unchanged part of the configuration.
- Primary test: two-sided Wilcoxon signed-rank test on paired F1 differences.
- Supplementary statistics: paired t-test, Shapiro-Wilk normality test on paired differences, and Cohen's dz.
- Multiple testing correction: Benjamini-Hochberg FDR within each generated test family.
- Significance threshold: FDR-adjusted p < 0.05.

Important identifiability note: NL appears in every semantic configuration. The CSVs do not contain a pure no-NL LLM/FTLM condition, so NL cannot be isolated as an individual semantic parameter. The report includes a separate GNN(R) versus GNN(FTLM(NL)) comparison, but that comparison is a semantic-embedding baseline comparison, not a pure NL-only ablation.

Link prediction uses bare semantic configuration names in the CSV. Following the requested interpretation, those rows are treated as `LLM` configurations in this report.

## Generated Plots

- `plots/significance_parameter_effects/k_vs0_effect_summary.png`: mean F1 change from k=0 to k=1,2,3.
- `plots/significance_parameter_effects/semantic_parameter_effect_summary.png`: effect of adding one semantic parameter to an otherwise matched configuration.
- `plots/significance_parameter_effects/combination_vs_nl_summary.png`: effect of semantic combinations compared with NL.
- `plots/significance_parameter_effects/combination_heatmap_<model>.png`: mean F1 by configuration and k.
- `plots/significance_parameter_effects/correlation_matrix_<model>.png`: Pearson correlations among semantic-parameter indicators, k, and F1.
- `plots/significance_parameter_effects/gnn_ftlm_vs_llm_matrix.png`: task-by-dataset significance matrix for `GNN(FTLM(config)) - LLM(config)`.
- `plots/significance_parameter_effects/significant_effect_counts.png`: counts of significant positive/negative effects.

## k-hop Effects

| model_type   |   k_to |   num_matched_tests |   mean_delta |   median_delta |   std_delta |   pvalue |   pvalue_uncorrected | significant_uncorrected_0.05   |   pvalue_fdr_bh | significant_fdr_0.05   |
|:-------------|-------:|--------------------:|-------------:|---------------:|------------:|---------:|---------------------:|:-------------------------------|----------------:|:-----------------------|
| GNN(FTLM)    |      1 |                  23 |       0.0522 |         0.0107 |      0.0924 |   0.0031 |               0.0031 | True                           |          0.0061 | True                   |
| GNN(FTLM)    |      2 |                  23 |       0.0397 |         0.0055 |      0.1211 |   0.2345 |               0.2345 | False                          |          0.2814 | False                  |
| GNN(FTLM)    |      3 |                  23 |       0.0083 |        -0.0077 |      0.1442 |   0.6010 |               0.6010 | False                          |          0.6010 | False                  |
| LLM          |      1 |                  30 |       0.0346 |         0.0235 |      0.0526 |   0.0000 |               0.0000 | True                           |          0.0001 | True                   |
| LLM          |      2 |                  30 |       0.0263 |         0.0236 |      0.0625 |   0.0234 |               0.0234 | True                           |          0.0351 | True                   |
| LLM          |      3 |                  30 |       0.0345 |         0.0247 |      0.0573 |   0.0002 |               0.0002 | True                           |          0.0006 | True                   |

## Individual Semantic Parameter Effects

| model_type   | parameter_added   |   num_matched_tests |   mean_delta |   median_delta |   std_delta |   pvalue |   pvalue_uncorrected | significant_uncorrected_0.05   |   pvalue_fdr_bh | significant_fdr_0.05   |
|:-------------|:------------------|--------------------:|-------------:|---------------:|------------:|---------:|---------------------:|:-------------------------------|----------------:|:-----------------------|
| GNN(FTLM)    | EL                |                  18 |       0.0059 |         0.0038 |      0.0294 |   0.5226 |               0.5226 | False                          |          0.7238 | False                  |
| GNN(FTLM)    | ET                |                  21 |       0.0175 |        -0.0006 |      0.0657 |   1.0000 |               1.0000 | False                          |          1.0000 | False                  |
| GNN(FTLM)    | NA                |                  28 |       0.0214 |         0.0118 |      0.0422 |   0.0103 |               0.0103 | True                           |          0.0821 | False                  |
| GNN(FTLM)    | NT                |                  36 |       0.0124 |         0.0054 |      0.0826 |   0.1189 |               0.1189 | False                          |          0.4756 | False                  |
| LLM          | EL                |                  18 |      -0.0124 |        -0.0015 |      0.0412 |   0.5509 |               0.5509 | False                          |          0.7238 | False                  |
| LLM          | ET                |                  21 |      -0.0000 |         0.0064 |      0.0277 |   0.6333 |               0.6333 | False                          |          0.7238 | False                  |
| LLM          | NA                |                  36 |      -0.0021 |         0.0033 |      0.0252 |   0.4051 |               0.4051 | False                          |          0.7238 | False                  |
| LLM          | NT                |                  47 |       0.0027 |         0.0015 |      0.0266 |   0.3415 |               0.3415 | False                          |          0.7238 | False                  |

## Combination Effects Compared With NL

| model_type   | config_to      |   num_matched_tests |   mean_delta |   median_delta |   std_delta |   pvalue |   pvalue_uncorrected | significant_uncorrected_0.05   |   pvalue_fdr_bh | significant_fdr_0.05   |
|:-------------|:---------------|--------------------:|-------------:|---------------:|------------:|---------:|---------------------:|:-------------------------------|----------------:|:-----------------------|
| GNN(FTLM)    | NL+NA          |                  28 |       0.0214 |         0.0118 |      0.0422 |   0.0103 |               0.0103 | True                           |          0.0499 | True                   |
| GNN(FTLM)    | NL+NA+NT       |                  25 |       0.0293 |         0.0141 |      0.0467 |   0.0125 |               0.0125 | True                           |          0.0499 | True                   |
| GNN(FTLM)    | NL+NA+NT+EL    |                  18 |       0.0269 |         0.0107 |      0.0452 |   0.0066 |               0.0066 | True                           |          0.0499 | True                   |
| GNN(FTLM)    | NL+NA+NT+EL+ET |                  12 |       0.0187 |        -0.0019 |      0.0637 |   1.0000 |               1.0000 | False                          |          1.0000 | False                  |
| LLM          | NL+NA          |                  36 |      -0.0021 |         0.0033 |      0.0252 |   0.4051 |               0.4051 | False                          |          0.4862 | False                  |
| LLM          | NL+NA+NT       |                  33 |      -0.0020 |         0.0057 |      0.0319 |   0.6720 |               0.6720 | False                          |          0.7331 | False                  |
| LLM          | NL+NA+NT+EL    |                  18 |      -0.0211 |        -0.0003 |      0.0469 |   0.3465 |               0.3465 | False                          |          0.4621 | False                  |
| LLM          | NL+NA+NT+EL+ET |                  12 |      -0.0417 |        -0.0082 |      0.0645 |   0.1763 |               0.1763 | False                          |          0.3022 | False                  |

## Omnibus Configuration Tests

| model_type   |   num_tests |   significant_fdr_0_05 |   median_pvalue_fdr |
|:-------------|------------:|-----------------------:|--------------------:|
| GNN(FTLM)    |          34 |                     30 |           0.0002063 |
| LLM          |          42 |                     34 |           0.0008268 |

## Correlation Matrix Insights

- `LLM`: strongest positive association with F1 is `NA` (r=0.27); weakest or most negative association is `ET` (r=-0.07).
- `GNN(FTLM)`: strongest positive association with F1 is `NA` (r=0.34); weakest or most negative association is `k` (r=0.03).

Correlation is descriptive rather than causal: the parameters are not fully orthogonal in the available configurations, so high parameter-parameter correlations indicate that some effects are coupled by the experimental design.

## GNN(FTLM) vs LLM Impact

| task                 | dataset    | scope           |    n |   mean_llm |   mean_gnn_ftlm |   delta |   pvalue_fdr_bh | significant_fdr_0.05   |
|:---------------------|:-----------|:----------------|-----:|-----------:|----------------:|--------:|----------------:|:-----------------------|
| Edge classification  | EAModelSet | task_dataset    |  110 |     0.8200 |          0.8460 |  0.0260 |          0.0017 | True                   |
| Edge classification  | Ecore-555  | task_dataset    |  180 |     0.9076 |          0.9380 |  0.0303 |          0.0000 | True                   |
| Edge classification  | ModelSet   | task_dataset    |  180 |     0.9292 |          0.9606 |  0.0314 |          0.0000 | True                   |
| Graph classification | Ecore-555  | task_dataset    |  180 |     0.9272 |          0.9525 |  0.0253 |          0.0000 | True                   |
| Graph classification | ModelSet   | task_dataset    |  180 |     0.7865 |          0.7795 | -0.0070 |          0.0001 | True                   |
| Node classification  | EAModelSet | task_dataset    |  200 |     0.7685 |          0.7707 |  0.0023 |          0.0229 | True                   |
| Node classification  | Ecore-555  | task_dataset    |  140 |     0.8933 |          0.9204 |  0.0271 |          0.0000 | True                   |
| Node classification  | ModelSet   | task_dataset    |  140 |     0.8832 |          0.8829 | -0.0003 |          0.3869 | False                  |
| Node classification  | OntoUML    | task_dataset    |  110 |     0.6878 |          0.7123 |  0.0244 |          0.0000 | True                   |
| Edge classification  | Overall    | task_overall    |  470 |     0.8954 |          0.9251 |  0.0297 |          0.0000 | True                   |
| Graph classification | Overall    | task_overall    |  360 |     0.8569 |          0.8660 |  0.0091 |          0.0030 | True                   |
| Node classification  | Overall    | task_overall    |  590 |     0.8103 |          0.8219 |  0.0117 |          0.0000 | True                   |
| Overall              | EAModelSet | dataset_overall |  310 |     0.7868 |          0.7974 |  0.0107 |          0.0000 | True                   |
| Overall              | Ecore-555  | dataset_overall |  500 |     0.9107 |          0.9383 |  0.0276 |          0.0000 | True                   |
| Overall              | ModelSet   | dataset_overall |  500 |     0.8650 |          0.8737 |  0.0087 |          0.0000 | True                   |
| Overall              | OntoUML    | dataset_overall |  110 |     0.6878 |          0.7123 |  0.0244 |          0.0000 | True                   |
| Overall              | Overall    | overall         | 1420 |     0.8503 |          0.8673 |  0.0170 |          0.0000 | True                   |

Overall, adding GNN on top of FTLM improves F1 by +0.0170 on average and is significant after FDR correction (q=1.902e-46, n=1420).

## NL Baseline Caveat

| model_type   | parameter_added   |   num_matched_tests |   mean_delta |   median_delta |   std_delta |   pvalue |   pvalue_uncorrected | significant_uncorrected_0.05   |   pvalue_fdr_bh | significant_fdr_0.05   |
|:-------------|:------------------|--------------------:|-------------:|---------------:|------------:|---------:|---------------------:|:-------------------------------|----------------:|:-----------------------|
| GNN(FTLM)    | NL                |                  16 |       0.4657 |         0.5361 |      0.2582 |   0.0000 |               0.0000 | True                           |          0.0000 | True                   |

This comparison should be read as GNN over NL FTLM embeddings versus random GNN features, not as a pure effect of node labels.

## Interpretation Guidance

- Positive delta means the second condition improved F1 over the first condition.
- A significant positive k result means adding k-hop information improved F1 over k=0 for that model family.
- A significant negative parameter or combination result means the added semantic information reduced F1 in the matched comparison.
- Because many effects differ by task and dataset, use the detailed CSV tables when making claims about a specific dataset or target.
