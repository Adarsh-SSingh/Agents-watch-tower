"""Research topic set for AgentOps data generation.

Deterministic (seeded). Each item: topic, category, difficulty, style.
Categories are chosen so the Research Agent produces different outcomes:
normal topics, broad/vague topics, very niche topics (few sources),
and nonsense topics (should fail or give insufficient sources).
"""

import random

TECH = [
    "retrieval augmented generation", "vector databases", "LLM agents",
    "transformer architecture", "model quantization", "LoRA fine-tuning",
    "prompt injection attacks", "Kubernetes autoscaling", "Rust vs Go",
    "WebAssembly", "edge computing", "federated learning",
    "diffusion models", "graph neural networks", "MLOps best practices",
    "LLM observability tools", "vLLM inference serving", "speculative decoding",
    "model context protocol", "serverless architecture", "zero trust security",
    "event driven architecture", "feature stores", "data drift detection",
]
SCIENCE = [
    "CRISPR gene editing", "mRNA vaccines", "nuclear fusion progress",
    "solid state batteries", "James Webb telescope discoveries",
    "quantum error correction", "microplastics in oceans",
    "perovskite solar cells", "gut microbiome research", "dark matter evidence",
    "room temperature superconductors", "lab grown meat",
    "carbon capture technology", "Alzheimer's drug trials", "exoplanet atmospheres",
]
BUSINESS = [
    "India startup funding trends", "UPI payment growth", "electric vehicle market in India",
    "semiconductor supply chain", "remote work productivity", "SaaS pricing models",
    "digital rupee CBDC", "gig economy regulation", "global inflation causes",
    "India renewable energy targets", "AI impact on jobs", "quick commerce competition",
    "open source business models", "ONDC e-commerce network", "fintech lending regulation",
]
HISTORY_SOCIETY = [
    "causes of the French Revolution", "history of the Silk Road", "Indus Valley civilization",
    "green revolution in India", "printing press impact", "industrial revolution in Britain",
    "history of the internet", "space race Cold War", "Mughal administration system",
    "impact of the Marshall Plan",
]
HEALTH = [
    "intermittent fasting evidence", "sleep and memory", "benefits of strength training",
    "ultra processed food risks", "mental health apps effectiveness", "vitamin D deficiency in India",
    "air pollution and health", "long COVID research",
]
NICHE = [
    "Kolmogorov-Arnold networks for time series", "Mamba state space models limitations",
    "history of the Jantar Mantar Jaipur", "Bhojpuri cinema economics",
    "mycelium based packaging", "Chandauli district agriculture",
    "tardigrade radiation resistance mechanism", "Zipf law in code identifiers",
]
VAGUE = [
    "AI", "future of technology", "best programming language", "health",
    "money", "the economy", "science news", "how to be successful",
]
NONSENSE = [
    "asdf qwerty zxcv", "blorptastic quantum flapdoodle theory",
    "the 2089 Mars election results", "secret recipe of the lost city of Zorgon",
]

TEMPLATES = {
    "plain": "{t}",
    "overview": "Give me an overview of {t}",
    "recent": "What are the latest developments in {t}?",
    "compare": "What are the main advantages and risks of {t}?",
    "hinglish": "{t} ke baare me detail me research karo",
    "explain": "Explain {t} with key facts and examples",
}

GROUPS = [
    ("technology", TECH, "medium"), ("science", SCIENCE, "medium"),
    ("business", BUSINESS, "medium"), ("history", HISTORY_SOCIETY, "easy"),
    ("health", HEALTH, "medium"), ("niche", NICHE, "hard"),
    ("vague", VAGUE, "hard"), ("nonsense", NONSENSE, "hard"),
]


def build_research_prompts(seed: int = 42, target: int = 200) -> list:
    rng = random.Random(seed)
    items = []
    seen = set()

    for category, subjects, difficulty in GROUPS:
        for subject in subjects:
            # vague and nonsense topics are used as written
            styles = ["plain"] if category in ("vague", "nonsense") else list(TEMPLATES)
            for style in styles:
                topic = TEMPLATES[style].format(t=subject)
                if topic in seen:
                    continue
                seen.add(topic)
                items.append({
                    "topic": topic, "category": category,
                    "difficulty": difficulty, "style": style, "subject": subject,
                })

    rng.shuffle(items)

    # Always keep every vague/nonsense topic: they are the rare failure cases.
    rare = [i for i in items if i["category"] in ("vague", "nonsense")]
    rest = [i for i in items if i["category"] not in ("vague", "nonsense")]
    chosen = rare + rest[:max(target - len(rare), 0)]
    rng.shuffle(chosen)
    return chosen[:target]