#!/usr/bin/env python3
"""
PsyDial 8-Dimension Counseling Skills Assessment (LLM Judge).
Evaluates TPPM-generated responses using PsyDial's official evaluation protocol.

Reference: PsyDial (Qiu & Lan, ACL 2025), Appendix J — Evaluation Metrics for Counseling Skills
Scoring protocol is EXACTLY aligned with PsyDial Table 3.
Judge model: DeepSeek-V4-Pro (configurable via DEEPSEEK_JUDGE_MODEL).

Environment:
    DEEPSEEK_API_KEY       — required
    DEEPSEEK_API_BASE      — optional (default: https://api.deepseek.com)
    DEEPSEEK_JUDGE_MODEL   — optional (default: deepseek-v4-pro)
"""

import json
import os
import time
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ============================================================
# Configuration — set via environment variables
# ============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
MODEL_NAME = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-pro")

# ============================================================
# Appendix J: 8 Counseling Skill Dimensions + OA
# Exact definitions and rating criteria from PsyDial paper
# ============================================================

# General rating scale description shared across all dimensions
RATING_SCALE_GENERAL = """
5-Point Likert Scale:
- 5: Excellent – Strong demonstration of the skill.
- 4: Good – Competent demonstration with room for minor improvement.
- 3: Average – Some strengths but notable gaps or inconsistencies.
- 2: Poor – Weak demonstration with several critical issues.
- 1: Very Poor – Significant issues with the skill, hindering the therapeutic process.
"""

# J.1 – J.8: Fine-grained counseling skill dimensions
METRICS_DEFINITIONS = {
    "empathy": {
        "name": "Empathy",
        "abbr": "Emp",
        "definition": (
            "Empathy refers to the counselor's ability to understand, resonate with, "
            "and validate the client's emotions and experiences. It involves not only "
            "recognizing the client's feelings but also communicating a deep sense of "
            "emotional understanding and support."
        ),
        "criteria": (
            "- 5: The counselor demonstrates deep empathy, consistently validating and "
            "responding to the client's emotions and experiences in a way that fosters connection.\n"
            "- 4: The counselor shows empathy, but it may occasionally lack depth or clarity in certain moments.\n"
            "- 3: The counselor shows basic empathy, but the emotional understanding feels somewhat distant or incomplete.\n"
            "- 2: The counselor struggles to demonstrate empathy, and emotional understanding feels superficial or lacking.\n"
            "- 1: The counselor does not show empathy or seems indifferent to the client's emotional experiences."
        ),
    },
    "active_listening": {
        "name": "Active Listening",
        "abbr": "AL",
        "definition": (
            "Active listening ensures a thorough understanding of the client's problems "
            "and emotions. The counselor attentively listens to both verbal and non-verbal "
            "cues, confirming the primary concerns and emotional state of the client. "
            "This helps build rapport and trust, while also making the client feel fully heard."
        ),
        "criteria": (
            "- 5: The counselor listens attentively without interruption, demonstrates full "
            "understanding, and accurately reflects the client's feelings and concerns.\n"
            "- 4: The counselor listens well but may occasionally miss small details or interrupt slightly.\n"
            "- 3: The counselor listens but struggles to pick up key details or misinterprets "
            "some aspects of the client's communication.\n"
            "- 2: The counselor listens partially, often missing important cues or failing "
            "to grasp the main concerns.\n"
            "- 1: The counselor does not listen actively, often interrupting or showing "
            "little engagement with the client's message."
        ),
    },
    "issue_clarification": {
        "name": "Issue Clarification",
        "abbr": "IC",
        "definition": (
            "Clarification involves seeking additional details or further explanation when "
            "the client's communication is unclear. The counselor asks specific questions to "
            "gain a better understanding of the client's situation, ensuring that all critical "
            "aspects of the problem are comprehended fully."
        ),
        "criteria": (
            "- 5: The counselor actively seeks clarification whenever necessary, asking precise "
            "questions that help unpack the client's issues clearly and comprehensively.\n"
            "- 4: The counselor seeks clarification in most situations, though some questions "
            "may be slightly general or unclear.\n"
            "- 3: The counselor asks some clarifying questions, but they may miss key aspects "
            "or fail to probe deeply enough.\n"
            "- 2: The counselor rarely asks for clarification, leaving gaps in understanding "
            "that could hinder progress.\n"
            "- 1: The counselor does not seek clarification, resulting in a poor understanding "
            "of the client's issues."
        ),
    },
    "open_ended_questioning": {
        "name": "Open-ended Questioning",
        "abbr": "OQ",
        "definition": (
            "Open-ended questions are designed to encourage the client to explore their "
            "thoughts, feelings, and experiences in greater depth. These questions usually "
            "start with 'how,' 'what,' or 'can you tell me more about,' and allow the client "
            "to provide expansive, reflective answers."
        ),
        "criteria": (
            "- 5: The counselor consistently uses open-ended questions that encourage deep "
            "exploration and self-reflection, promoting rich, meaningful dialogue.\n"
            "- 4: The counselor asks open-ended questions but may sometimes rely on closed "
            "or leading questions.\n"
            "- 3: The counselor occasionally uses open-ended questions but often defaults "
            "to yes/no questions or questions with limited scope.\n"
            "- 2: The counselor uses very few open-ended questions, limiting the client's "
            "ability to explore their own thoughts and feelings.\n"
            "- 1: The counselor avoids open-ended questions entirely, only asking yes/no "
            "or directive questions."
        ),
    },
    "encouraging_self_exploration": {
        "name": "Encouraging Self-Exploration",
        "abbr": "ESE",
        "definition": (
            "Encouraging self-exploration means asking questions and providing prompts that "
            "help the client reflect on their own emotions, thoughts, behaviors, and "
            "decision-making. This promotes greater self-awareness and empowers the client "
            "to make their own insights and choices."
        ),
        "criteria": (
            "- 5: The counselor frequently encourages the client to explore their own thoughts "
            "and feelings, fostering significant self-awareness and insight.\n"
            "- 4: The counselor encourages self-exploration but may not consistently prompt "
            "the client to explore deeper layers of their experiences.\n"
            "- 3: The counselor occasionally encourages self-exploration, but it may lack "
            "depth or clarity, limiting the client's reflection.\n"
            "- 2: The counselor provides limited opportunities for self-exploration, directing "
            "the conversation more than encouraging self-reflection.\n"
            "- 1: The counselor does not encourage self-exploration, instead providing "
            "solutions or interpretations without engaging the client's own thoughts."
        ),
    },
    "cognitive_restructuring": {
        "name": "Cognitive Restructuring",
        "abbr": "CR",
        "definition": (
            "Cognitive restructuring involves helping the client identify and challenge "
            "distorted or unrealistic thought patterns. The counselor assists the client in "
            "reframing negative or maladaptive thoughts, fostering more realistic and helpful "
            "cognitive patterns that promote emotional well-being."
        ),
        "criteria": (
            "- 5: The counselor skillfully helps the client identify distorted thoughts and "
            "gently guides them to more balanced, realistic perspectives.\n"
            "- 4: The counselor helps challenge distorted thinking, but may not consistently "
            "provide clear alternatives or insight.\n"
            "- 3: The counselor offers some cognitive restructuring, but the process feels "
            "incomplete or lacks sufficient exploration of thought patterns.\n"
            "- 2: The counselor rarely engages in cognitive restructuring, providing minimal "
            "guidance for challenging negative thoughts.\n"
            "- 1: The counselor does not address cognitive distortions or fails to help the "
            "client change unhelpful thinking patterns."
        ),
    },
    "guided_questioning": {
        "name": "Guided Questioning",
        "abbr": "GQ",
        "definition": (
            "Guided questioning refers to the use of focused questions to help the client "
            "narrow down their concerns or focus on specific goals. This approach helps the "
            "client clarify their thoughts and find solutions to specific problems, often by "
            "prompting deeper reflection on particular aspects of their experience."
        ),
        "criteria": (
            "- 5: The counselor uses guided questions effectively, helping the client focus "
            "on specific issues or goals in a way that enhances clarity and progress.\n"
            "- 4: The counselor uses guided questions but may not always focus them as "
            "effectively on the client's immediate needs or goals.\n"
            "- 3: The counselor uses some guiding questions, but they may be overly broad "
            "or fail to narrow in on the client's main concerns.\n"
            "- 2: The counselor rarely uses guiding questions, or their questions lack focus, "
            "making it difficult for the client to concentrate on specific issues.\n"
            "- 1: The counselor does not use guiding questions, and the session lacks focus "
            "or clarity on specific goals."
        ),
    },
    "non_judgmental_accepting_attitude": {
        "name": "Non-judgmental and Accepting Attitude",
        "abbr": "NJAA",
        "definition": (
            "A non-judgmental and accepting attitude means creating a safe and supportive "
            "environment where the client feels free to share their thoughts and feelings "
            "without fear of criticism or negative judgment. The counselor maintains a neutral, "
            "respectful approach, accepting the client's experiences and emotional expressions "
            "without imposing their own values or opinions."
        ),
        "criteria": (
            "- 5: The counselor consistently maintains a non-judgmental, accepting stance, "
            "allowing the client to share openly without fear of judgment.\n"
            "- 4: The counselor is generally non-judgmental, with occasional lapses in "
            "maintaining complete neutrality or acceptance.\n"
            "- 3: The counselor maintains a neutral attitude but may unintentionally come "
            "across as judgmental in some instances.\n"
            "- 2: The counselor's attitude is occasionally critical or dismissive, potentially "
            "creating discomfort for the client.\n"
            "- 1: The counselor is overtly judgmental or dismissive, making the client feel "
            "unsafe or unsupported."
        ),
    },
    "overall_assessment": {
        "name": "Overall Assessment",
        "abbr": "OA",
        "definition": (
            "This overall score combines the key evaluation principles into one holistic "
            "rating scale. The counselor's performance will be evaluated based on how "
            "effectively they apply each principle in their practice. Each principle contributes "
            "equally to the final score, providing a comprehensive assessment of the counselor's "
            "abilities in understanding and responding to the client's needs. The overall score "
            "will reflect the counselor's skill in fostering a supportive, non-judgmental, and "
            "effective therapeutic environment."
        ),
        "evaluation_principles": (
            "- Active Listening: Ensure a complete understanding of the client's issues and "
            "emotions, while confirming the client's main concerns and feelings.\n"
            "- Empathy: Express understanding and care for the client's emotions. Use "
            "empathetic language such as 'I understand how you feel' or 'It sounds like "
            "you're really sad.'\n"
            "- Issue Clarification: If the client's communication is unclear, the counselor "
            "can ask specific questions to ensure a full understanding of their situation.\n"
            "- Open-ended Questions: The counselor can use open-ended questions to encourage "
            "the client to provide more information and elaborate on their thoughts and feelings.\n"
            "- Encouraging Self-Exploration: The counselor can ask questions that encourage "
            "the client to explore their feelings, thoughts, and behaviors in order to promote "
            "self-reflection.\n"
            "- Cognitive Restructuring: Help the client identify and challenge unrealistic or "
            "distorted thought patterns, guiding them toward more balanced thinking.\n"
            "- Guided Questioning: The counselor can use guiding questions to help the client "
            "focus on specific issues or goals, clarifying their thoughts and moving toward "
            "resolution.\n"
            "- Non-judgmental and Accepting Attitude: Avoid making judgments about the "
            "client's experiences or emotions. Use neutral language and respect the client's "
            "perspectives and choices."
        ),
        "criteria": (
            "- 5: Excellent – The counselor consistently demonstrates mastery of all principles. "
            "Their approach is empathetic, insightful, and highly effective in addressing the "
            "client's needs. The counselor maintains a strong, non-judgmental rapport while "
            "promoting self-awareness and growth.\n"
            "- 4: Good – The counselor effectively applies the principles in most areas. There "
            "may be occasional lapses in one or two principles, but overall the counselor's "
            "approach is competent, and the client feels supported and understood.\n"
            "- 3: Average – The counselor demonstrates adequate skills in applying the principles. "
            "However, there are noticeable gaps or inconsistencies in their practice. Some areas "
            "may need improvement to enhance the therapeutic process.\n"
            "- 2: Below Average – The counselor struggles to effectively apply several principles. "
            "There are significant gaps in understanding or responding to the client's needs, "
            "resulting in limited therapeutic progress. The approach may sometimes feel "
            "disconnected or judgmental.\n"
            "- 1: Poor – The counselor demonstrates minimal or no proficiency in applying the "
            "principles. Their responses are ineffective, and they may create an unhelpful or "
            "even detrimental therapeutic environment. The client is likely to feel unsupported "
            "or misunderstood."
        ),
    },
}

# Dimension order for output
DIMENSION_KEYS = [
    "empathy", "active_listening", "issue_clarification",
    "open_ended_questioning", "encouraging_self_exploration",
    "cognitive_restructuring", "guided_questioning",
    "non_judgmental_accepting_attitude", "overall_assessment",
]

# ============================================================
# Prompt Builder — exactly follows PsyDial Figure 19 template
# ============================================================

SYSTEM_PROMPT = (
    "You are a professional psychological counseling supervisor. "
    "You should evaluate how well the counselor applies their skills "
    "in a counseling context."
)


def format_dialogue_history(messages: List[Dict]) -> str:
    """Format the dialogue history messages into a readable text."""
    lines = []
    for msg in messages:
        role_label = "来访者" if msg["role"] == "user" else "咨询师"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


def format_metric_text(key: str) -> str:
    """Format a single metric definition + criteria, exactly as PsyDial Appendix J.

    This is the text injected as {metric} in PsyDial Figure 19's get_rating_prompt().
    """
    m = METRICS_DEFINITIONS[key]
    text = f"""{m['name']} ({m['abbr']})

Definition: {m['definition']}

"""
    # OA (J.9) has Evaluation Principles between Definition and Rating Criteria
    if key == "overall_assessment" and "evaluation_principles" in m:
        text += f"""Evaluation Principles:
{m['evaluation_principles']}

"""

    text += f"""Rating Criteria:
{m['criteria']}"""
    return text


def build_scoring_prompt(ctx: str, response: str, metric_key: str) -> str:
    """Build scoring prompt for ONE metric — exactly matches PsyDial Figure 19.

    Args:
        ctx: Formatted dialogue history text
        response: Counselor's generated response
        metric_key: One of DIMENSION_KEYS

    Returns:
        User prompt string matching Figure 19 template.
    """
    metric_text = format_metric_text(metric_key)

    # Exact template from PsyDial Figure 19
    prompt = f"""The following is a counseling context.
Dialogue history: {ctx}
Counselor's response: {response}
{metric_text}

Provide a brief reasoning for your rating based on these criteria, and then assign a numerical rating. Provide your answer in the following format.

- Reasoning: (Your explanation here)
- Rating: (Ranging from 1 to 5)"""
    return prompt


# ============================================================
# LLM Judge API Client
# ============================================================

def call_judge(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Optional[Dict]:
    """Call LLM Judge API (DeepSeek-V4-Pro) with retry logic."""
    import requests

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(30, 90),
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # Try to parse JSON from response
                scores = parse_response_json(content)
                if scores:
                    return scores
                # If parse failed, retry
                print(f"  Warning: JSON parse failed, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"  Timeout, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"  Error: {e}, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return None


def parse_response_json(content: str) -> Optional[Dict]:
    """Parse the PsyDial Figure 19 output format:

    - Reasoning: (explanation text)
    - Rating: (integer 1-5)

    Returns dict like {'reasoning': '...', 'rating': 4} or None on failure.
    """
    reasoning = ""
    rating = None

    # Match "- Reasoning: ..." — may span multiple lines until "- Rating:" is found
    # Use regex to extract
    import re

    # Try to find reasoning text (everything between "- Reasoning:" and "- Rating:")
    reason_match = re.search(r'- Reasoning:\s*(.+?)(?=\n- Rating:|\Z)', content, re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    # Find the rating number
    rating_match = re.search(r'- Rating:\s*\(?Ranging from \d+ to \d+\)?\s*(\d+)', content)
    if not rating_match:
        # Try simpler pattern: "- Rating: N"
        rating_match = re.search(r'- Rating:\s*(\d+)', content)
    if rating_match:
        rating = int(rating_match.group(1))

    if rating is not None and 1 <= rating <= 5:
        return {"reasoning": reasoning, "rating": rating}

    return None


# ============================================================
# Main Scoring Pipeline
# ============================================================

def load_generations(filepath: str) -> Dict:
    """Load TPPM generation results."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_d101_data(filepath: str) -> Dict[int, List[Dict]]:
    """Load D101 dataset and index by idx for dialogue history lookup.

    Returns: dict mapping idx -> messages list
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["idx"]: item["messages"] for item in data}


def load_checkpoint(output_path: str) -> Dict[int, Dict]:
    """Load existing scores from a checkpoint file."""
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["idx"]: item for item in data.get("scores", [])}


def save_checkpoint(output_path: str, metadata: Dict, scores: List[Dict]):
    """Save current progress."""
    judge_info = {
        "judge_provider": "DeepSeek",
        "judge_model": MODEL_NAME,
        "judge_api_base": API_BASE,
    }
    output = {
        "metadata": {**metadata, **judge_info},
        "scores": sorted(scores, key=lambda x: x["idx"]),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def compute_summary(scores: List[Dict]) -> Dict:
    """Compute mean and std for each dimension."""
    summary = {}
    for key in DIMENSION_KEYS:
        ratings = [s[key]["rating"] for s in scores]
        summary[key] = {
            "mean": sum(ratings) / len(ratings) if ratings else 0,
            "std": (sum((r - sum(ratings)/len(ratings))**2 for r in ratings) / len(ratings)) ** 0.5 if ratings else 0,
            "count": len(ratings),
        }
    # Overall OA mean
    oa_ratings = [s["overall_assessment"]["rating"] for s in scores]
    summary["overall_assessment"]["mean"] = sum(oa_ratings) / len(oa_ratings) if oa_ratings else 0
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge 8-Dimension Counseling Skills Evaluation (DeepSeek-V4-Pro, PsyDial Protocol Appendix J)"
    )
    parser.add_argument(
        "--generations",
        type=str,
        default="/root/autodl-tmp/wangqihao/Table1-data/outputs/eval/d101_full/tppm_memory_generations_v2.json",
        help="Path to TPPM generation results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/wangqihao/Table1-data/outputs/eval/d101_full/tppm_memory_judge_scores.json",
        help="Output path for LLM Judge scores",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First case index to score (for resuming)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Last case index to score (exclusive, for testing subset)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--d101",
        type=str,
        default="/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D101/PsyDial-D101.json",
        help="Path to PsyDial-D101.json for dialogue history lookup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without making API calls",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading generations from: {args.generations}")
    gen_data = load_generations(args.generations)
    results = gen_data["results"]
    total = len(results)
    print(f"Total cases: {total}")
    print(f"Metadata: {json.dumps(gen_data['metadata'], ensure_ascii=False)}")

    # Load D101 dialogue histories
    print(f"Loading D101 dialogue histories from: {args.d101}")
    d101_messages = load_d101_data(args.d101)
    print(f"D101 cases loaded: {len(d101_messages)}")

    # Load checkpoint
    existing_scores_map = load_checkpoint(args.output)
    print(f"Existing scores in checkpoint: {len(existing_scores_map)}")

    # Determine range
    end_idx = args.end_idx if args.end_idx is not None else total
    end_idx = min(end_idx, total)

    # Prepare scores list
    scores_list = list(existing_scores_map.values())

    # Count statistics
    fallback_count = sum(1 for r in results if "fallback_reason" in r)
    valid_count = total - fallback_count
    print(f"Fallback cases: {fallback_count} (no TPPM memory used)")
    print(f"Valid cases (TPPM memory used): {valid_count}")

    # Dry run: show first case prompts for all 9 metrics
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — Printing prompts for first case (all 9 metrics)")
        print("=" * 60)
        case = results[0]
        dialogue_msgs = d101_messages.get(case["idx"], [])
        ctx = format_dialogue_history(dialogue_msgs) if dialogue_msgs else "[No messages]"
        print(f"\n[SYSTEM]\n{SYSTEM_PROMPT}\n")
        for key in DIMENSION_KEYS:
            prompt = build_scoring_prompt(ctx, case["generated"], key)
            print(f"\n--- {METRICS_DEFINITIONS[key]['name']} ---")
            print(prompt[:1200])
            print("...")
        return

    # Score each case — PsyDial protocol: 9 separate API calls per case
    # Each call evaluates ONE dimension independently
    scored = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i in range(args.start_idx, end_idx):
        case = results[i]
        idx = case["idx"]

        # Skip if already fully scored
        existing = existing_scores_map.get(idx, {})
        if all(key in existing for key in DIMENSION_KEYS):
            skipped += 1
            continue

        # Format dialogue context from D101 dataset
        dialogue_msgs = d101_messages.get(idx, [])
        ctx = format_dialogue_history(dialogue_msgs) if dialogue_msgs else "[Dialogue history not found in D101]"
        response = case["generated"]
        golden = case.get("golden", "")

        if (i + 1) % 5 == 0 or i == args.start_idx:
            elapsed = time.time() - start_time
            rate = (scored + 1) / max(elapsed, 1) * 60
            print(f"\nCase {i + 1}/{total} | Scored: {scored} | Skipped: {skipped} | "
                  f"Errors: {errors} | Rate: {rate:.1f}/min | Elapsed: {elapsed:.0f}s")

        # Score each of the 9 dimensions with a separate API call (PsyDial protocol)
        scores_for_case = {}
        case_failed = False
        for metric_key in DIMENSION_KEYS:
            # Skip if this metric already scored in checkpoint
            if metric_key in existing:
                scores_for_case[metric_key] = existing[metric_key]
                continue

            user_prompt = build_scoring_prompt(ctx, response, metric_key)
            api_result = call_judge(SYSTEM_PROMPT, user_prompt)

            if api_result is None:
                print(f"  FAILED: idx={idx}, metric={METRICS_DEFINITIONS[metric_key]['abbr']}")
                errors += 1
                case_failed = True
                # Record as missing
                scores_for_case[metric_key] = {"reasoning": "API_FAILED", "rating": 0}
            else:
                scores_for_case[metric_key] = api_result

            time.sleep(args.delay)

        # Build score entry
        score_entry = {
            "idx": idx,
            "golden": golden,
            "generated": response,
            "fallback_reason": case.get("fallback_reason"),
        }
        for key in DIMENSION_KEYS:
            score_entry[key] = scores_for_case[key]

        scores_list.append(score_entry)
        if not case_failed:
            scored += 1

        # Print brief result
        ratings = [f"{METRICS_DEFINITIONS[k]['abbr']}={scores_for_case[k]['rating']}"
                   for k in DIMENSION_KEYS]
        fb = f" [fallback: {case['fallback_reason']}]" if case.get("fallback_reason") else ""
        print(f"  idx={idx} | {' '.join(ratings)}{fb}" + (" [PARTIAL]" if case_failed else ""))

        # Save checkpoint every 10 cases (since each case does 9 calls = ~90 calls per checkpoint)
        if scored % 10 == 0 and scored > 0:
            save_checkpoint(args.output, gen_data["metadata"], scores_list)
            print(f"  [Checkpoint saved: {len(scores_list)} scores]")

    # Final save
    save_checkpoint(args.output, gen_data["metadata"], scores_list)
    print(f"\n{'=' * 60}")
    print(f"Scoring complete. Total scored: {scored}, Skipped: {skipped}, Errors: {errors}")
    print(f"Output: {args.output}")

    # Compute and print summary
    if scores_list:
        summary = compute_summary(scores_list)
        print(f"\n{'=' * 60}")
        print("SUMMARY (Mean Scores)")
        print(f"{'=' * 60}")
        print(f"{'Dimension':<35} {'Abbr':<6} {'Mean':>6}  {'Std':>6}  {'N':>5}")
        print("-" * 62)
        for key in DIMENSION_KEYS:
            m = METRICS_DEFINITIONS[key]
            s = summary[key]
            print(f"{m['name']:<35} {m['abbr']:<6} {s['mean']:6.3f}  {s['std']:6.3f}  {s['count']:5d}")

        # Print in LaTeX table format
        print(f"\n{'=' * 60}")
        print("LaTeX Table Row (for tab:exp1_layer1_auto)")
        print(f"{'=' * 60}")
        tex_values = " & ".join(
            f"{summary[k]['mean']:.2f}" for k in DIMENSION_KEYS
        )
        print(f"\\methodname{{}} (本文) & {tex_values} \\\\")

        # Separate fallback vs valid summary
        valid_scores = [s for s in scores_list if not s.get("fallback_reason")]
        fallback_scores = [s for s in scores_list if s.get("fallback_reason")]
        if valid_scores and fallback_scores:
            valid_summary = compute_summary(valid_scores)
            fb_summary = compute_summary(fallback_scores)
            print(f"\nValid (TPPM memory): OA mean = {valid_summary['overall_assessment']['mean']:.3f} (N={len(valid_scores)})")
            print(f"Fallback:           OA mean = {fb_summary['overall_assessment']['mean']:.3f} (N={len(fallback_scores)})")


if __name__ == "__main__":
    main()
