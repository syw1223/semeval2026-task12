import os
# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import random
import uuid
from pathlib import Path
from typing import List, Dict

from transformers import pipeline
from tqdm import tqdm

# =============== 配置 ===============
# 原始 questions.jsonl 的路径（按你项目实际调整）
INPUT_QUEST_PATH = "data/train/questions.jsonl"

# 增强后的保存路径（不会覆盖原始文件）
OUTPUT_QUEST_PATH = "data/train/questions_augmented_1000.jsonl"
DEVICE = 1
# 目标总 question 数量（原始 + 合成 >= 这个数）
TARGET_QUESTION_COUNT = 1000

# 使用的改写模型（可以换成你能下到的 paraphrase 模型）
PARA_MODEL_NAME = "ramsrigouthamg/t5_paraphraser"


# 每条原始 question 最多生成多少个变体（上限）
VARIANTS_PER_QUESTION = 10

# 每个 target_event 生成几个 paraphrase
EVENT_PARAPHRASES_PER_QUESTION = 2

# 每个选项生成几个 paraphrase
OPTION_PARAPHRASES_PER_OPTION = 1


# =============== 工具函数 ===============

def load_questions(path: str) -> List[Dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def save_questions(path: str, questions: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def build_paraphraser():
    print(f"Loading paraphrase model: {PARA_MODEL_NAME}")
    para = pipeline(
        "text2text-generation",
        model=PARA_MODEL_NAME,
        device=DEVICE,
    )
    return para


def paraphrase_n(para, text: str, n: int, max_length: int = 64) -> List[str]:
    """调用大模型生成 n 个改写版本，去重并过滤掉和原文几乎一样的。"""
    text = (text or "").strip()
    if n <= 0 or not text:
        return []

    prompt = f"paraphrase: {text}"
    outputs = para(
        prompt,
        max_length=max_length,
        num_return_sequences=n,
        do_sample=True,
        num_beams=max(4, n),
        temperature=1.0,
    )

    cands = []
    text_norm = text.lower()
    for out in outputs:
        # 对于 text2text-generation，键一般是 "generated_text"
        t = out.get("generated_text", "").strip().strip('"').strip()
        if not t:
            continue
        if t.lower() == text_norm:
            continue
        if t in cands:
            continue
        cands.append(t)
    return cands


# =============== 主逻辑 ===============

def main():
    # 1. 读原始 questions
    input_path = Path(INPUT_QUEST_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Input questions file not found: {input_path}")

    orig_questions = load_questions(str(input_path))
    print(f"原始 questions 数量: {len(orig_questions)}")

    existing_count = len(orig_questions)
    if existing_count >= TARGET_QUESTION_COUNT:
        print("原始 questions 已经 >= 目标数量，无需增强。")
        output_path = Path(OUTPUT_QUEST_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_questions(str(output_path), orig_questions)
        print(f"已将原始 questions 直接保存为: {output_path}")
        return

    num_synthetic_needed = TARGET_QUESTION_COUNT - existing_count
    print(f"需要合成的 question 数量: {num_synthetic_needed}")

    # 2. 加载 paraphrase 大模型
    paraphraser = build_paraphraser()

    synthetic_questions: List[Dict] = []
    global_index = 0

    # 3. 遍历原始 questions，逐个生成变体
    #    为了保证够 1000 条，可以循环多轮，直到 synthetic 数量够
    with tqdm(total=num_synthetic_needed, desc="Generating synthetic questions") as pbar:
        while len(synthetic_questions) < num_synthetic_needed:
            for q in orig_questions:
                if len(synthetic_questions) >= num_synthetic_needed:
                    break

                topic_id = q.get("topic_id")
                target_event = q.get("target_event", "")
                options = {
                    "A": q.get("option_A", ""),
                    "B": q.get("option_B", ""),
                    "C": q.get("option_C", ""),
                    "D": q.get("option_D", ""),
                }
                golden_answer = q.get("golden_answer", "A")
                base_uuid = q.get("uuid", "")

                # 3.1 为 target_event 生成若干 paraphrase
                event_candidates = [target_event]
                try:
                    event_paras = paraphrase_n(
                        paraphraser,
                        target_event,
                        EVENT_PARAPHRASES_PER_QUESTION,
                        max_length=64,
                    )
                    event_candidates.extend(event_paras)
                except Exception as e:
                    print(f"[WARN] event paraphrase failed for uuid={base_uuid}: {e}")

                # 3.2 为每个 option 生成 paraphrase 候选
                option_candidates = {}
                for opt_label, opt_text in options.items():
                    cands = [opt_text]
                    try:
                        paras = paraphrase_n(
                            paraphraser,
                            opt_text,
                            OPTION_PARAPHRASES_PER_OPTION,
                            max_length=48,
                        )
                        cands.extend(paras)
                    except Exception as e:
                        print(f"[WARN] option {opt_label} paraphrase failed for uuid={base_uuid}: {e}")
                    option_candidates[opt_label] = cands

                # 3.3 组合生成若干个变体（保持 golden_answer 不变）
                for _ in range(VARIANTS_PER_QUESTION):
                    if len(synthetic_questions) >= num_synthetic_needed:
                        break

                    new_event = random.choice(event_candidates)
                    new_opt_A = random.choice(option_candidates["A"])
                    new_opt_B = random.choice(option_candidates["B"])
                    new_opt_C = random.choice(option_candidates["C"])
                    new_opt_D = random.choice(option_candidates["D"])

                    new_q = {
                        "topic_id": topic_id,
                        "uuid": f"synthetic-{uuid.uuid4()}",
                        "target_event": new_event,
                        "option_A": new_opt_A,
                        "option_B": new_opt_B,
                        "option_C": new_opt_C,
                        "option_D": new_opt_D,
                        "golden_answer": golden_answer,
                    }
                    synthetic_questions.append(new_q)
                    global_index += 1
                    pbar.update(1)
                    if len(synthetic_questions) >= num_synthetic_needed:
                        break

    # 4. 合并原始 + 合成，并保存
    all_questions = orig_questions + synthetic_questions
    print(f"增强后 questions 总数: {len(all_questions)}")

    output_path = Path(OUTPUT_QUEST_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_questions(str(output_path), all_questions)
    print(f"已保存增强后的 questions 到: {output_path}")


if __name__ == "__main__":
    main()
