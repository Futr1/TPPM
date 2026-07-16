# Experiment setup writing style notes, 2026-07-01

Purpose: Compare experiment setup structures in related long-term memory, user profiling, and mental-health dialogue papers, then assess the structure of `draft/TPPM-draft.tex`.

Sources checked:
- Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL 2024. https://aclanthology.org/2024.acl-long.747/
- Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. COLM 2025 / arXiv. https://arxiv.org/abs/2504.14225 and https://arxiv.org/html/2504.14225v1
- PsyDial: A Large-scale Long-term Conversational Dataset for Mental Health Counseling. ACL 2025. Local PDF: `/root/autodl-tmp/wangqihao/实验参考文献/PsyDial.pdf`; ACL page: https://aclanthology.org/2025.acl-long.1049/
- MemoryART: Enhancing LLMs via Multi-Memory Models with Adaptive Resonance Theory for Healthcare Agents. Local PDF: `/root/autodl-tmp/wangqihao/实验参考文献/AAAI/MemoryART.pdf`
- A-Mem: Agentic Memory for LLM Agents. Local PDF: `/root/autodl-tmp/wangqihao/实验参考文献/A-Mem.pdf`; OpenReview page: https://openreview.net/forum?id=FiM0M8gcct

Observed structures:
- LoCoMo: The benchmark construction section defines tasks first; the experimental setup is split by task/model family. It states model categories (base, long-context, RAG), task-specific constraints, metrics, and implementation details in appendix.
- PersonaMem: The experiment narrative is driven by research questions and query types. It introduces the benchmark setting, model families, query categories, discriminative/generative evaluation settings, RAG variants, and length/recency analyses.
- PsyDial: The section uses a highly explicit hierarchy: Experiments -> Experimental Setups -> Backbone Models -> Model Training -> Baselines -> Evaluation Set -> Evaluations. This structure is clear for mental-health dialogue work because it separates model training, baselines, test set construction, and evaluation protocol.
- MemoryART: The experiment setup is concise and uses paragraph labels: Dataset, Evaluation Metrics, and then results. It is suitable when the method is evaluated on two benchmarks and the setup is not the main contribution.
- A-Mem: The experiment section is compact: Dataset and Evaluation, Implementation Details, Empirical Results. It combines dataset, baselines, and metrics in one subsection, then isolates reproducibility-related implementation choices.

Diagnosis for the current manuscript:
- Current Section 4 has the right high-level components: datasets, metrics, baselines.
- It is too compressed for a three-benchmark paper because each benchmark corresponds to a different capability and evaluation protocol.
- It does not explicitly state research questions/evaluation goals before datasets.
- It does not isolate implementation details, hyperparameters, backbone/base LLM, retrieval top-k, judge model/settings, and randomization/stability controls in the main setup section.
- The PsyDial-like mental-health evaluation would benefit from an explicit `Evaluation Protocol` subsection, including Likert scale, dimensions, judge model, and reliability/consistency handling.
