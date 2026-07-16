#!/usr/bin/env python3
"""Score Golden responses from D101 using the same deepseek-chat scoring protocol.

This produces a direct, same-scorer comparison: TPPM vs Golden.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
import json, os, time, re, argparse
import os
from typing import Dict, List, Optional

# ============================================================
# Config — same as llm_judge_scoring.py
# ============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )
API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
MODEL_NAME = "deepseek-v4-flash"

SYSTEM_PROMPT = (
    "You are a professional psychological counseling supervisor. "
    "You should evaluate how well the counselor applies their skills "
    "in a counseling context."
)

METRICS_DEFINITIONS = {
    "empathy": {
        "name": "Empathy", "abbr": "Emp",
        "definition": "Empathy refers to the counselor's ability to understand, resonate with, and validate the client's emotions and experiences. It involves not only recognizing the client's feelings but also communicating a deep sense of emotional understanding and support.",
        "criteria": "- 5: The counselor demonstrates deep empathy, consistently validating and responding to the client's emotions and experiences in a way that fosters connection.\n- 4: The counselor shows empathy, but it may occasionally lack depth or clarity in certain moments.\n- 3: The counselor shows basic empathy, but the emotional understanding feels somewhat distant or incomplete.\n- 2: The counselor struggles to demonstrate empathy, and emotional understanding feels superficial or lacking.\n- 1: The counselor does not show empathy or seems indifferent to the client's emotional experiences.",
    },
    "active_listening": {
        "name": "Active Listening", "abbr": "AL",
        "definition": "Active listening ensures a thorough understanding of the client's problems and emotions. The counselor attentively listens to both verbal and non-verbal cues, confirming the primary concerns and emotional state of the client. This helps build rapport and trust, while also making the client feel fully heard.",
        "criteria": "- 5: The counselor listens attentively without interruption, demonstrates full understanding, and accurately reflects the client's feelings and concerns.\n- 4: The counselor listens well but may occasionally miss small details or interrupt slightly.\n- 3: The counselor listens but struggles to pick up key details or misinterprets some aspects of the client's communication.\n- 2: The counselor listens partially, often missing important cues or failing to grasp the main concerns.\n- 1: The counselor does not listen actively, often interrupting or showing little engagement with the client's message.",
    },
    "issue_clarification": {
        "name": "Issue Clarification", "abbr": "IC",
        "definition": "Clarification involves seeking additional details or further explanation when the client's communication is unclear. The counselor asks specific questions to gain a better understanding of the client's situation, ensuring that all critical aspects of the problem are comprehended fully.",
        "criteria": "- 5: The counselor actively seeks clarification whenever necessary, asking precise questions that help unpack the client's issues clearly and comprehensively.\n- 4: The counselor seeks clarification in most situations, though some questions may be slightly general or unclear.\n- 3: The counselor asks some clarifying questions, but they may miss key aspects or fail to probe deeply enough.\n- 2: The counselor rarely asks for clarification, leaving gaps in understanding that could hinder progress.\n- 1: The counselor does not seek clarification, resulting in a poor understanding of the client's issues.",
    },
    "open_ended_questioning": {
        "name": "Open-ended Questioning", "abbr": "OQ",
        "definition": "Open-ended questions are designed to encourage the client to explore their thoughts, feelings, and experiences in greater depth. These questions usually start with 'how,' 'what,' or 'can you tell me more about,' and allow the client to provide expansive, reflective answers.",
        "criteria": "- 5: The counselor consistently uses open-ended questions that encourage deep exploration and self-reflection, promoting rich, meaningful dialogue.\n- 4: The counselor asks open-ended questions but may sometimes rely on closed or leading questions.\n- 3: The counselor occasionally uses open-ended questions but often defaults to yes/no questions or questions with limited scope.\n- 2: The counselor uses very few open-ended questions, limiting the client's ability to explore their own thoughts and feelings.\n- 1: The counselor avoids open-ended questions entirely, only asking yes/no or directive questions.",
    },
    "encouraging_self_exploration": {
        "name": "Encouraging Self-Exploration", "abbr": "ESE",
        "definition": "Encouraging self-exploration means asking questions and providing prompts that help the client reflect on their own emotions, thoughts, behaviors, and decision-making. This promotes greater self-awareness and empowers the client to make their own insights and choices.",
        "criteria": "- 5: The counselor frequently encourages the client to explore their own thoughts and feelings, fostering significant self-awareness and insight.\n- 4: The counselor encourages self-exploration but may not consistently prompt the client to explore deeper layers of their experiences.\n- 3: The counselor occasionally encourages self-exploration, but it may lack depth or clarity, limiting the client's reflection.\n- 2: The counselor provides limited opportunities for self-exploration, directing the conversation more than encouraging self-reflection.\n- 1: The counselor does not encourage self-exploration, instead providing solutions or interpretations without engaging the client's own thoughts.",
    },
    "cognitive_restructuring": {
        "name": "Cognitive Restructuring", "abbr": "CR",
        "definition": "Cognitive restructuring involves helping the client identify and challenge distorted or unrealistic thought patterns. The counselor assists the client in reframing negative or maladaptive thoughts, fostering more realistic and helpful cognitive patterns that promote emotional well-being.",
        "criteria": "- 5: The counselor skillfully helps the client identify distorted thoughts and gently guides them to more balanced, realistic perspectives.\n- 4: The counselor helps challenge distorted thinking, but may not consistently provide clear alternatives or insight.\n- 3: The counselor offers some cognitive restructuring, but the process feels incomplete or lacks sufficient exploration of thought patterns.\n- 2: The counselor rarely engages in cognitive restructuring, providing minimal guidance for challenging negative thoughts.\n- 1: The counselor does not address cognitive distortions or fails to help the client change unhelpful thinking patterns.",
    },
    "guided_questioning": {
        "name": "Guided Questioning", "abbr": "GQ",
        "definition": "Guided questioning refers to the use of focused questions to help the client narrow down their concerns or focus on specific goals. This approach helps the client clarify their thoughts and find solutions to specific problems, often by prompting deeper reflection on particular aspects of their experience.",
        "criteria": "- 5: The counselor uses guided questions effectively, helping the client focus on specific issues or goals in a way that enhances clarity and progress.\n- 4: The counselor uses guided questions but may not always focus them as effectively on the client's immediate needs or goals.\n- 3: The counselor uses some guiding questions, but they may be overly broad or fail to narrow in on the client's main concerns.\n- 2: The counselor rarely uses guiding questions, or their questions lack focus, making it difficult for the client to concentrate on specific issues.\n- 1: The counselor does not use guiding questions, and the session lacks focus or clarity on specific goals.",
    },
    "non_judgmental_accepting_attitude": {
        "name": "Non-judgmental and Accepting Attitude", "abbr": "NJAA",
        "definition": "A non-judgmental and accepting attitude means creating a safe and supportive environment where the client feels free to share their thoughts and feelings without fear of criticism or negative judgment. The counselor maintains a neutral, respectful approach, accepting the client's experiences and emotional expressions without imposing their own values or opinions.",
        "criteria": "- 5: The counselor consistently maintains a non-judgmental, accepting stance, allowing the client to share openly without fear of judgment.\n- 4: The counselor is generally non-judgmental, with occasional lapses in maintaining complete neutrality or acceptance.\n- 3: The counselor maintains a neutral attitude but may unintentionally come across as judgmental in some instances.\n- 2: The counselor's attitude is occasionally critical or dismissive, potentially creating discomfort for the client.\n- 1: The counselor is overtly judgmental or dismissive, making the client feel unsafe or unsupported.",
    },
    "overall_assessment": {
        "name": "Overall Assessment", "abbr": "OA",
        "definition": "This overall score combines the key evaluation principles into one holistic rating scale. The counselor's performance will be evaluated based on how effectively they apply each principle in their practice. Each principle contributes equally to the final score, providing a comprehensive assessment of the counselor's abilities in understanding and responding to the client's needs. The overall score will reflect the counselor's skill in fostering a supportive, non-judgmental, and effective therapeutic environment.",
        "evaluation_principles": "- Active Listening: Ensure a complete understanding of the client's issues and emotions, while confirming the client's main concerns and feelings.\n- Empathy: Express understanding and care for the client's emotions. Use empathetic language such as 'I understand how you feel' or 'It sounds like you're really sad.'\n- Issue Clarification: If the client's communication is unclear, the counselor can ask specific questions to ensure a full understanding of their situation.\n- Open-ended Questions: The counselor can use open-ended questions to encourage the client to provide more information and elaborate on their thoughts and feelings.\n- Encouraging Self-Exploration: The counselor can ask questions that encourage the client to explore their feelings, thoughts, and behaviors in order to promote self-reflection.\n- Cognitive Restructuring: Help the client identify and challenge unrealistic or distorted thought patterns, guiding them toward more balanced thinking.\n- Guided Questioning: The counselor can use guiding questions to help the client focus on specific issues or goals, clarifying their thoughts and moving toward resolution.\n- Non-judgmental and Accepting Attitude: Avoid making judgments about the client's experiences or emotions. Use neutral language and respect the client's perspectives and choices.",
        "criteria": "- 5: Excellent – The counselor consistently demonstrates mastery of all principles. Their approach is empathetic, insightful, and highly effective in addressing the client's needs. The counselor maintains a strong, non-judgmental rapport while promoting self-awareness and growth.\n- 4: Good – The counselor effectively applies the principles in most areas. There may be occasional lapses in one or two principles, but overall the counselor's approach is competent, and the client feels supported and understood.\n- 3: Average – The counselor demonstrates adequate skills in applying the principles. However, there are noticeable gaps or inconsistencies in their practice. Some areas may need improvement to enhance the therapeutic process.\n- 2: Below Average – The counselor struggles to effectively apply several principles. There are significant gaps in understanding or responding to the client's needs, resulting in limited therapeutic progress. The approach may sometimes feel disconnected or judgmental.\n- 1: Poor – The counselor demonstrates minimal or no proficiency in applying the principles. Their responses are ineffective, and they may create an unhelpful or even detrimental therapeutic environment. The client is likely to feel unsupported or misunderstood.",
    },
}

DIMENSION_KEYS = [
    "empathy", "active_listening", "issue_clarification",
    "open_ended_questioning", "encouraging_self_exploration",
    "cognitive_restructuring", "guided_questioning",
    "non_judgmental_accepting_attitude", "overall_assessment",
]

def format_dialogue_history(messages):
    lines = []
    for msg in messages:
        role_label = "来访者" if msg["role"] == "user" else "咨询师"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)

def format_metric_text(key):
    m = METRICS_DEFINITIONS[key]
    text = f"{m['name']} ({m['abbr']})\n\nDefinition: {m['definition']}\n\n"
    if key == "overall_assessment" and "evaluation_principles" in m:
        text += f"Evaluation Principles:\n{m['evaluation_principles']}\n\n"
    text += f"Rating Criteria:\n{m['criteria']}"
    return text

def build_scoring_prompt(ctx, response, metric_key):
    metric_text = format_metric_text(metric_key)
    return f"""The following is a counseling context.
Dialogue history: {ctx}
Counselor's response: {response}
{metric_text}

Provide a brief reasoning for your rating based on these criteria, and then assign a numerical rating. Provide your answer in the following format.

- Reasoning: (Your explanation here)
- Rating: (Ranging from 1 to 5)"""

def call_api(system_prompt, user_prompt, max_retries=3):
    import requests
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], "temperature": 0.0, "max_tokens": 4096}

    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=(30, 90))
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = parse_response(content)
                if parsed:
                    return parsed
                if attempt < max_retries - 1:
                    time.sleep(2)
            elif resp.status_code == 429:
                time.sleep(min(2 ** attempt * 5, 60))
            else:
                if attempt < max_retries - 1:
                    time.sleep(5)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5)
    return None

def parse_response(content):
    reason_match = re.search(r'- Reasoning:\s*(.+?)(?=\n- Rating:|\Z)', content, re.DOTALL)
    reasoning = reason_match.group(1).strip() if reason_match else ""
    rating_match = re.search(r'- Rating:\s*\(?Ranging from \d+ to \d+\)?\s*(\d+)', content)
    if not rating_match:
        rating_match = re.search(r'- Rating:\s*(\d+)', content)
    if rating_match:
        rating = int(rating_match.group(1))
        if 1 <= rating <= 5:
            return {"reasoning": reasoning, "rating": rating}
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d101", default="/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json")
    parser.add_argument("--output", default="REPO_ROOT / 'benchmarks/psydial'/outputs/eval/d101_full/golden_scores_v2.json")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.d101) as f:
        d101 = json.load(f)

    # Load checkpoint
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = {s["idx"]: s for s in json.load(f).get("scores", [])}

    scores_list = list(existing.values())
    end_idx = min(args.end_idx, len(d101))

    scored = 0
    for i in range(args.start_idx, end_idx):
        case = d101[i]
        idx = case["idx"]

        if idx in existing and all(k in existing[idx] for k in DIMENSION_KEYS):
            scored += 1
            continue

        msgs = case["messages"][:-1]  # exclude golden from context
        golden_content = case["golden"]["content"]
        ctx = format_dialogue_history(msgs)

        scores_for_case = {}
        for metric_key in DIMENSION_KEYS:
            if metric_key in existing.get(idx, {}):
                scores_for_case[metric_key] = existing[idx][metric_key]
                continue
            prompt = build_scoring_prompt(ctx, golden_content, metric_key)
            result = call_api(SYSTEM_PROMPT, prompt)
            if result is None:
                scores_for_case[metric_key] = {"reasoning": "API_FAILED", "rating": 0}
            else:
                scores_for_case[metric_key] = result
            time.sleep(args.delay)

        entry = {"idx": idx, "golden": golden_content}
        for k in DIMENSION_KEYS:
            entry[k] = scores_for_case[k]
        scores_list.append(entry)
        scored += 1

        ratings_str = " ".join(f"{METRICS_DEFINITIONS[k]['abbr']}={scores_for_case[k]['rating']}" for k in DIMENSION_KEYS)
        print(f"[{scored}/{end_idx - args.start_idx}] idx={idx} | {ratings_str}")

        if scored % 10 == 0:
            with open(args.output, "w") as f:
                json.dump({"metadata": {"scorer": MODEL_NAME, "target": "golden"}, "scores": sorted(scores_list, key=lambda x: x["idx"])}, f, ensure_ascii=False, indent=2)
            print(f"  [Checkpoint: {len(scores_list)}]")

    with open(args.output, "w") as f:
        json.dump({"metadata": {"scorer": MODEL_NAME, "target": "golden"}, "scores": sorted(scores_list, key=lambda x: x["idx"])}, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\nDone. {len(scores_list)} golden responses scored.")
    import statistics
    for k in DIMENSION_KEYS:
        ratings = [s[k]["rating"] for s in scores_list if s[k]["rating"] > 0]
        if ratings:
            print(f"  {METRICS_DEFINITIONS[k]['abbr']}: mean={statistics.mean(ratings):.2f} std={statistics.stdev(ratings):.2f}")

if __name__ == "__main__":
    main()
