import os
# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import random
from transformers import pipeline

# ================== 配置 ==================
# 原始 docs.json 路径（改成你自己的）
INPUT_DOCS_PATH = "data/train/docs.json"
# 增强后保存路径
OUTPUT_DOCS_PATH = "data/train/docs_augmented_1000.json"

TARGET_DOCS_COUNT = 1000  # 目标总新闻条数

# ================== 1. 读取原始数据 ==================
with open(INPUT_DOCS_PATH, "r", encoding="utf-8") as f:
    docs_data = json.load(f)

# docs_data 结构类似：
# [
#   {
#       "topic_id": 1,
#       "topic": "...",
#       "docs": [ {title, link, snippet, source, imageUrl, content, uuid}, ... ]
#   },
#   ...
# ]

# 统计原始新闻总数
original_docs = []
for topic in docs_data:
    for doc in topic["docs"]:
        original_docs.append({
            "topic_id": topic["topic_id"],
            "topic": topic["topic"],
            "doc": doc,
        })

existing_news_count = len(original_docs)
print(f"原始新闻条数: {existing_news_count}")

num_synthetic_needed = max(0, TARGET_DOCS_COUNT - existing_news_count)
print(f"需要生成的合成新闻条数: {num_synthetic_needed}")

if num_synthetic_needed == 0:
    print("原始数据已经 >= 目标数量，不需要增强。")
    exit(0)

# ================== 2. 初始化生成模型（GPT-2） ==================
# 如果你只有一块 GPU：
generator = pipeline(
    "text-generation",
    model="gpt2-medium",
    device=1  # 用 cuda:1
)

# ================== 3. 定义增强函数 ==================
def generate_synthetic_docs(original_docs, num_synthetic_samples):
    synthetic_docs_by_topic = {}  # topic_id -> list[new_doc]

    for i in range(num_synthetic_samples):
        # 随机抽一条原始新闻作为“种子”
        seed = random.choice(original_docs)
        topic_id = seed["topic_id"]
        topic_name = seed["topic"]
        src_doc = seed["doc"]

        # 为了多样化，设计几种不同的 prompt 模板
        prompt_templates = [
            (
                "You are a journalist. Based on the following article about \"{topic}\", "
                "write a new news article from a different angle, focusing on international reactions "
                "and possible future implications.\n\nOriginal article:\n{content}\n\nNew article:"
            ),
            (
                "Write a detailed news article about the topic \"{topic}\". "
                "The article should be inspired by the following text, "
                "but do NOT copy sentences. Emphasize political context and EU-level debates.\n\n"
                "Reference text:\n{content}\n\nNew article:"
            ),
            (
                "Imagine you are reporting a follow-up story one month after the following event "
                "on the topic \"{topic}\". Describe what has happened since, including new developments, "
                "public opinion, and government responses.\n\nPrevious report:\n{content}\n\nFollow-up article:"
            ),
        ]

        # 随机选一个模板
        template = random.choice(prompt_templates)
        seed_content = src_doc.get("content", "")
        # 为了防止 prompt 太长，可以截断一点原文
        seed_content_short = seed_content[:1200]

        prompt = template.format(topic=topic_name, content=seed_content_short)

        # 生成新新闻
        out = generator(
            prompt,
            max_new_tokens=256,       # 控制生成长度
            num_return_sequences=1,
            do_sample=True,           # 采样生成，增加多样性
            temperature=0.9,
            top_p=0.95
        )[0]["generated_text"]

        # 把 prompt 之前的部分去掉，只保留“新文章”那一段，简单粗暴一点
        # 找最后一次出现 "New article:" 或 "Follow-up article:" 之后的部分
        split_tokens = ["New article:", "Follow-up article:"]
        clean_text = out
        for tok in split_tokens:
            if tok in out:
                clean_text = out.split(tok, 1)[-1].strip()
        if not clean_text:
            clean_text = out.strip()

        # 构造一个新的 doc 结构，尽量保持和原始 docs.json 格式一致
        new_doc = {
            "title": f"[Synthetic] {src_doc.get('title', '')}",
            "link": "",                 # 合成的，没有真实链接
            "snippet": clean_text[:200].replace("\n", " "),
            "source": "synthetic_gpt2",
            "imageUrl": "",
            "content": clean_text,
            "uuid": f"synthetic-{topic_id}-{i}"
        }

        synthetic_docs_by_topic.setdefault(topic_id, []).append(new_doc)

    return synthetic_docs_by_topic

# ================== 4. 生成合成新闻 ==================
synthetic_docs_by_topic = generate_synthetic_docs(original_docs, num_synthetic_needed)

# ================== 5. 把合成新闻插回原始结构 ==================
augmented_docs_data = []
for topic in docs_data:
    tid = topic["topic_id"]
    new_topic = {
        "topic_id": topic["topic_id"],
        "topic": topic["topic"],
        "docs": list(topic["docs"])  # 先放原始的
    }
    extra_docs = synthetic_docs_by_topic.get(tid, [])
    new_topic["docs"].extend(extra_docs)
    augmented_docs_data.append(new_topic)

# 再次统计总数确认
final_count = sum(len(t["docs"]) for t in augmented_docs_data)
print(f"增强后新闻总数: {final_count}")

# ================== 6. 保存增强后的 docs ==================
with open(OUTPUT_DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(augmented_docs_data, f, ensure_ascii=False, indent=2)

print(f"已保存到: {OUTPUT_DOCS_PATH}")
