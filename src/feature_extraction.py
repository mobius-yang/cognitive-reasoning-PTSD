# feature_extraction.py
# 根据标准格式的数据进行写作内容特征提取，参照VAM/SAM的标准对于文本进行打分，用于后续模型训练

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pandas as pd
from tqdm import tqdm

MAPPING = {
        "temporal": {"clear": 3, "mixed": 2, "unclear": 1},          # 时间线
        "coherence": {"coherent": 3, "mixed": 2, "fragmented": 1},   # 连贯性
        "reflection": {"present": 3, "minimal": 2, "absent": 1},     # 情绪表达
        "sensory": {"low": 1, "medium": 2, "high": 3},               # 感官细节
        "arousal": {"low": 1, "medium": 2, "high": 3},               # 情绪强度
        "language": {"past":3, "present":1},                         # 语言形式
        "perspective": {"narrator":3, "re-experiencer":1}            # 体验特征
}

DIMENSIONS = {
    "temporal": ["clear", "mixed", "unclear"],
    "coherence": ["coherent", "mixed", "fragmented"],
    "reflection": ["present", "minimal", "absent"],
    "sensory": ["low", "medium", "high"],
    "arousal": ["low", "medium", "high"],
    "language": ["past", "present"],
    "perspective": ["narrator", "re-experiencer"]
}

def build_single_prompt(text, dim, options):
    prompt = f"""
    You are a clinical psychology researcher.
    Analyze the following trauma narrative.

    Narrative:{text}

    Question:
    For the dimension "{dim}", choose ONE option.

    Options:
    {options}

    Answer with ONLY ONE word from the options.
    """
    return prompt

def parse_vam_sam_with_criteria(text, llm_pipeline):
    
    raw_ans = {}
    scores = {k: 0 for k in MAPPING.keys()}

    for dim, options in DIMENSIONS.items():
        prompt = build_single_prompt(text, dim, options)
        output = llm_pipeline(prompt, max_length=50, do_sample=False)[0]["generated_text"].lower().strip()
        for word, score in MAPPING[dim].items():
            if word in output:
                scores[dim] = score
                raw_ans[dim] = word
                break

    
    # 这里可以进行分数计算的更改，属于heuristic部分
    vam_score = (
        scores["temporal"] +
        scores["coherence"] +
        scores["reflection"] +
        scores["language"] +
        scores["perspective"]
    ) / 5.0

    sam_score = (
        scores["sensory"] +
        scores["arousal"]
    ) / 2.0

    if vam_score > sam_score:
        dominant = "VAM"
    elif sam_score > vam_score:
        dominant = "SAM"
    else:
        dominant = "Balanced"

    return {
        "temporal": scores["temporal"],
        "coherence": scores["coherence"],
        "reflection": scores["reflection"],
        "sensory": scores["sensory"],
        "arousal": scores["arousal"],
        "language": scores["language"],
        "perspective": scores["perspective"],
        "VAM_index": vam_score,
        "SAM_index": sam_score,
        "dominant": dominant
    }


if __name__ == "__main__":
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    df = pd.read_excel("./cleaned_data.xlsx")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row["narrative"]

        if not isinstance(text, str) or len(text.strip()) < 5:
            results.append({k: 0 for k in list(MAPPING.keys()) + ["VAM_index", "SAM_index"]} | {"dominant": None})
            continue
        
        try:
            scores = parse_vam_sam_with_criteria(text, llm_pipeline)
            results.append(scores)
        except Exception as e:
            print(f"Error processing row {_}: {e}")
            results.append({
                "temporal": 0,
                "coherence": 0,
                "reflection": 0,
                "sensory": 0,
                "arousal": 0,
                "language": 0,
                "perspective": 0,
                "VAM_index": 0,
                "SAM_index": 0,
                "dominant": None
            })

    score_df = pd.DataFrame(results)
    df_final = pd.concat([df, score_df], axis=1)
    df_final.to_excel("./task1_scored.xlsx", index=False)

