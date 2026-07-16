# Abstract style lookup notes, 2026-07-01

Purpose: Check whether related CCF-A / high-quality AI papers typically explain ablation studies at the end of the abstract, and gather examples for revising `draft/TPPM-draft.tex`.

Sources checked:
- Generative Agents: Interactive Simulacra of Human Behavior. arXiv / UIST 2023. https://arxiv.org/abs/2304.03442
- Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL 2024. https://aclanthology.org/2024.acl-long.747/
- Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. COLM 2025 / arXiv. https://arxiv.org/abs/2504.14225
- Can AI Relate: Testing Large Language Model Response for Mental Health Support. Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.120/
- PsyDial: A Large-scale Long-term Conversational Dataset for Mental Health Counseling. ACL 2025. https://aclanthology.org/2025.acl-long.1049/
- AGILE: A Novel Reinforcement Learning Framework of LLM Agents. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/097c514162ea7126d40671d23e12f51b-Abstract-Conference.html

Observed pattern:
- LoCoMo, PersonaMem, PsyDial, and the EMNLP mental-health evaluation paper foreground problem motivation, research gap, proposed dataset/framework, evaluation setting, and main findings. Their abstracts do not explain ablation results in detail.
- Generative Agents and AGILE include a brief final ablation sentence, but only as a concise supporting result for the proposed system.
- For this manuscript, the abstract should not spend its final sentence explaining each ablated component. A cleaner ending is to state the main empirical advantage and leave ablation evidence to the experiment section or contribution list.
