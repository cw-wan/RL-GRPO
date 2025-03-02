# Enhancing LLM's Reasoning Ability with Rule-based GRPO

A final project for CSC_52081_EP - Apprentissage Automatique Avancé et Agents Autonomes (2024-2025).

## Guide

Train GRPO:

```shell
accelerate launch train.py
```

Run evaluation:

```shell
./evaluate.sh
```

## Results

### GSM8K

|Model| Solve Rate |
|-----|------------|
|Qwen2.5-0.5B-Instruct| 35.61 |

## Reference

[1] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y. and Guo, D., 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

[2] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X. and Zhang, X., 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.
