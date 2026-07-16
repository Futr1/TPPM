# Contribution-writing lookup notes, 2026-07-01

Purpose: Compare introduction-ending contribution paragraphs in related long-term memory, user profiling, and mental-health dialogue papers, then evaluate the contribution list in `draft/TPPM-draft.tex`.

Sources checked:
- Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023 / arXiv. https://dl.acm.org/doi/10.1145/3586183.3606763 and https://arxiv.org/abs/2304.03442
- Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL 2024. https://aclanthology.org/2024.acl-long.747/ and https://aclanthology.org/2024.acl-long.747.pdf
- Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. COLM 2025 / arXiv. https://arxiv.org/abs/2504.14225 and https://arxiv.org/html/2504.14225v1
- Can AI Relate: Testing Large Language Model Response for Mental Health Support. Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.120/
- Do Large Language Models Align with Core Mental Health Counseling Competencies? Findings of NAACL 2025. https://aclanthology.org/2025.findings-naacl.418.pdf
- PsyDial: A Large-scale Long-term Conversational Dataset for Mental Health Counseling. ACL 2025. https://aclanthology.org/2025.acl-long.1049/

Observed contribution-writing patterns:
- Contributions are usually concrete deliverables: a new architecture/framework, a benchmark/dataset/evaluation protocol, empirical findings, released code/data, or safety/ethical analysis.
- Strong papers avoid making each bullet a long restatement of method details. They state the artifact and why it matters.
- Dataset/benchmark papers often use contribution bullets to separate: resource construction, evaluation framework, model/baseline findings, and release.
- Method/system papers often use contribution bullets to separate: system/task framing, architecture, evaluation evidence, and limitation/risk analysis.
- Mental-health papers often make safety, ethics, professional standards, or human oversight visible in the intro/contribution framing when the claim touches counseling deployment.

Preliminary diagnosis for this manuscript:
- The current first and second bullets overlap: both describe the TPPM architecture/mechanisms.
- The theoretical contribution is mixed with implementation details; it is not clear whether the novelty is a psychological framing, a memory unit, or the update/retrieval algorithm.
- The third bullet is broad and resembles a results summary; it does not specify the evaluation contribution sharply enough.
- The contribution list does not explicitly mention safety/clinical boundary or responsible use, which is important for mental-health support papers.
- Some wording still reads like a method description rather than an externally verifiable contribution.
