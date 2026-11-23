# Study materials

Features and driving results for the 40 PCLA agents from 18 families, evaluated on the same 220 Bench2Drive routes.

| File or folder | Contents |
| --- | --- |
| [Agents Feature Catalogue.xlsx](Agents%20Feature%20Catalogue.xlsx) | Definitions of the 51 features, explanations of their values, the characteristics of each agent and its overall results. |
| [Agents Analysis.xlsx](Agents%20Analysis.xlsx) | Statistical analysis of Driving Score, with regression results, prediction errors, comparisons across towns and related agents, and charts. |
| [Bench2Drive Results.xlsx](Bench2Drive%20Results.xlsx) | Further results by agent, family, traffic situation, town and weather, plus infractions and feature comparisons. |
| [benchmark_results/](benchmark_results/) | One JSON per agent, named with its PCLA identifier, containing the final results for all 220 routes. |
| [figures/](figures/) | Figures illustrating the study results and procedure. |
| [analysis/](analysis/) | Python code and input descriptions for repeating the statistical calculations. |

The catalogue's **Feature_Dictionary** sheet explains the recorded characteristics.

## Run the analysis

With Python 3.12, run these commands from this folder:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
```

The code reads the catalogue and benchmark results, then saves the numerical calculations in `calculated_results/`:

- `agent_data.csv`: the analysis variables and driving measures for each agent.
- `models.csv`, `coefficients.csv` and `lofo_predictions.csv`: regression results, 95% confidence intervals and leave-one-family-out predictions.
- `prediction_groups.csv`, `variant_comparisons.csv` and `town13_comparisons.csv`: comparisons within CARLA-version groups, between related agents, and between variants trained with or without Town13.
