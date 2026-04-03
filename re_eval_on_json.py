import json
import ollama

# ============ CONFIG ============
INPUT_JSON  = "./gpt_eval_comparison_20B_cleaned.json"
OUTPUT_JSON = "./gpt_eval_reeval_results.json"
JUDGE_MODEL = "llama3.1:8b"
MODEL_KEYS  = ["your_model", "1t_model"]

# ============ OLLAMA EVAL FUNCTION ============
def gpt_eval(prompt, story, model_name=JUDGE_MODEL):
    eval_prompt = f"""the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
    The exercise tests the student's language abilities and creativity. 
    The symbol *** marks the separator between the prescribed beginning and the student's completion: {prompt}***{story}

    Please provide your general assessment about the part written by the student (the one after the *** symbol). Is it grammatically correct? Is it consistent with the beginning of the story? 
    Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***. 

    Now, grade the student's completion in terms of grammar (1-10), creativity (1-10), consistency (1-10), plot (1-10).

    Return ONLY valid JSON and nothing else — no explanation, no markdown, no backticks:
    {{"grammar": 1, "creativity": 1, "consistency": 1, "plot": 1}}"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": eval_prompt}],
        format="json",
        options={"temperature": 0.0},
    )

    raw = response["message"]["content"]
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    scores = json.loads(raw)

    for key in ["grammar", "creativity", "consistency", "plot"]:
        scores[key] = max(1, min(10, int(scores[key])))

    scores["overall"] = round(
        (scores["grammar"] + scores["creativity"] + scores["consistency"] + scores["plot"]) / 4, 2
    )
    return scores

# ============ RE-EVALUATE ONE MODEL ============
def reevaluate_model(model_data, model_key):
    print(f"\n{'='*60}")
    print(f"Re-evaluating: {model_key}")
    print(f"{'='*60}")

    all_grammar     = []
    all_creativity  = []
    all_consistency = []
    all_plot        = []
    all_overall     = []
    new_per_prompt  = []

    per_prompt = model_data["per_prompt"]

    for i, entry in enumerate(per_prompt):
        prompt  = entry["prompt"]
        stories = entry["stories"]
        n       = len(stories)

        print(f"\nPrompt {i+1}/{len(per_prompt)} ({n} stories): '{prompt[:80]}...'")

        prompt_grammar     = []
        prompt_creativity  = []
        prompt_consistency = []
        prompt_plot        = []

        for j, story in enumerate(stories):
            try:
                scores = gpt_eval(prompt, story)
                prompt_grammar.append(scores["grammar"])
                prompt_creativity.append(scores["creativity"])
                prompt_consistency.append(scores["consistency"])
                prompt_plot.append(scores["plot"])

                print(
                    f"  [{j+1}/{n}] "
                    f"grammar={scores['grammar']} "
                    f"creativity={scores['creativity']} "
                    f"consistency={scores['consistency']} "
                    f"plot={scores['plot']} "
                    f"overall={scores['overall']}"
                )
            except Exception as e:
                print(f"  [{j+1}/{n}] ERROR: {e} — skipping story")

        if not prompt_grammar:
            print("  No valid scores for this prompt, skipping.")
            continue

        n_valid = len(prompt_grammar)
        avg_grammar     = round(sum(prompt_grammar)     / n_valid, 2)
        avg_creativity  = round(sum(prompt_creativity)  / n_valid, 2)
        avg_consistency = round(sum(prompt_consistency) / n_valid, 2)
        avg_plot        = round(sum(prompt_plot)        / n_valid, 2)
        avg_overall     = round((avg_grammar + avg_creativity + avg_consistency + avg_plot) / 4, 2)

        all_grammar.append(avg_grammar)
        all_creativity.append(avg_creativity)
        all_consistency.append(avg_consistency)
        all_plot.append(avg_plot)
        all_overall.append(avg_overall)

        new_per_prompt.append({
            "prompt"          : prompt,
            "stories"         : stories,
            "avg_grammar"     : avg_grammar,
            "avg_creativity"  : avg_creativity,
            "avg_consistency" : avg_consistency,
            "avg_plot"        : avg_plot,
            "avg_overall"     : avg_overall,
        })

        print(
            f"  → Prompt average: "
            f"grammar={avg_grammar} creativity={avg_creativity} "
            f"consistency={avg_consistency} plot={avg_plot} overall={avg_overall}"
        )

    n_prompts = len(all_grammar)
    return {
        "model"       : model_data.get("model", model_key),
        "grammar"     : round(sum(all_grammar)     / n_prompts, 2),
        "creativity"  : round(sum(all_creativity)  / n_prompts, 2),
        "consistency" : round(sum(all_consistency) / n_prompts, 2),
        "plot"        : round(sum(all_plot)        / n_prompts, 2),
        "overall"     : round(sum(all_overall)     / n_prompts, 2),
        "per_prompt"  : new_per_prompt,
    }

# ============ MAIN ============
with open(INPUT_JSON) as f:
    data = json.load(f)

output = {"evaluation_config": data.get("evaluation_config", {})}
output["evaluation_config"]["evaluator"] = JUDGE_MODEL
output["evaluation_config"]["note"] = "Re-evaluation of existing stories from cleaned JSON"

for model_key in MODEL_KEYS:
    if model_key not in data:
        print(f"WARNING: '{model_key}' not found in JSON, skipping.")
        continue
    output[model_key] = reevaluate_model(data[model_key], model_key)

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_JSON}")

# ============ PRINT FINAL COMPARISON ============
print(f"\n{'='*65}")
print("FINAL COMPARISON")
print(f"{'='*65}")
print(f"\n{'Metric':<15}", end="")
for key in MODEL_KEYS:
    print(f" {key:>20}", end="")
print()
print("─" * (15 + 21 * len(MODEL_KEYS)))

for metric in ["grammar", "creativity", "consistency", "plot", "overall"]:
    print(f"{metric:<15}", end="")
    for key in MODEL_KEYS:
        val = output.get(key, {}).get(metric, "N/A")
        print(f" {str(val):>20}", end="")
    print()