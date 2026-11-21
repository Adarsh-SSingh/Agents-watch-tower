"""Prompt set for the Tool Agent (used to generate bulk run data).

Every prompt has:
- prompt          : text sent to the agent
- category        : kind of task (used later as an ML feature / for analysis)
- expected_tools  : tools the agent should call (empty = no tool needed)
- difficulty      : easy | medium | hard
- style           : how the prompt is worded (plain, polite, verbose, ...)
"""

import random

CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad", "Pune",
    "Jaipur", "Lucknow", "Chandigarh", "Kochi", "Ahmedabad", "London", "Paris",
    "Tokyo", "New York", "Sydney", "Dubai", "Singapore", "Berlin", "Toronto",
    "Cape Town", "Moscow", "Seoul", "Rome", "Madrid", "Bangkok", "Cairo",
    "Nairobi", "Lima",
]

TOPICS = [
    "LangChain", "Transformer (machine learning model)", "Random forest",
    "Gradient descent", "Retrieval-augmented generation", "Large language model",
    "Neural network", "Python (programming language)", "Docker (software)",
    "Kubernetes", "PostgreSQL", "SQLite", "Streamlit", "FastAPI", "Pandas (software)",
    "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Reinforcement learning",
    "Convolutional neural network", "Support vector machine", "K-means clustering",
    "Overfitting", "Backpropagation", "Attention (machine learning)",
    "Vector database", "Word embedding", "Isaac Newton", "Albert Einstein",
    "Alan Turing", "Ada Lovelace", "Bengaluru", "Photosynthesis", "Black hole",
    "Moon landing", "Industrial Revolution", "Internet", "Blockchain", "Cloud computing",
]

# Topics / cities that do not exist: the tool should fail or return nothing.
BAD_CITIES = ["Xyzzyville", "Qwertyland", "Zzzzburg", "Nowhereton"]
BAD_TOPICS = ["Flibbertigibbet protocol 9000", "Zorblax theory of snacks", "Qwxz language model"]

GENERAL_QUESTIONS = [
    "Explain what an API is in two lines.",
    "What is the difference between a list and a tuple in Python?",
    "Give me three tips for writing clean code.",
    "What does HTTP stand for?",
    "Explain overfitting in one sentence.",
    "What is the capital of Japan?",
    "Write a one-line definition of machine learning.",
    "Why is the sky blue? Answer briefly.",
    "What is a REST API? Keep it short.",
    "Suggest a name for a pet goldfish.",
]


def _item(prompt, category, tools, difficulty):
    return {
        "prompt": prompt,
        "category": category,
        "expected_tools": tools,
        "difficulty": difficulty,
        "style": "plain",
    }


def _build_base_prompts(seed: int = 42) -> list:
    rng = random.Random(seed)
    items = []

    # ---------------- calculator ----------------
    for _ in range(30):
        a, b = rng.randint(2, 999), rng.randint(2, 999)
        op, word = rng.choice([("+", "plus"), ("-", "minus"), ("x", "times")])
        items.append(_item(f"What is {a} {op} {b}?", "calculator", ["calculator"], "easy"))
    for _ in range(15):
        n = rng.randint(2, 40) ** 2
        items.append(_item(f"What is the square root of {n}?", "calculator", ["calculator"], "easy"))
    for _ in range(15):
        p, r, y = rng.choice([10000, 25000, 50000, 120000]), rng.choice([5, 6, 7, 8, 10]), rng.randint(2, 6)
        items.append(_item(
            f"Calculate the compound interest on {p} at {r}% per year for {y} years.",
            "calculator", ["calculator"], "medium"))
    for _ in range(10):
        n = rng.randint(5, 15)
        items.append(_item(f"What is {n} factorial?", "calculator", ["calculator"], "medium"))
    for _ in range(10):
        a, b = rng.randint(10, 500), rng.randint(2, 60)
        items.append(_item(f"What is {b}% of {a * 10}?", "calculator", ["calculator"], "easy"))
    for _ in range(8):
        a, b = rng.randint(2, 30), rng.randint(2, 9)
        items.append(_item(
            f"Calculate {a} to the power {b}, then take its square root.",
            "calculator", ["calculator"], "medium"))
    # calculator failure cases
    for text in [
        "Calculate 100 divided by 0.",
        "What is the square root of -25?",
        "Calculate the logarithm of 0.",
        "What is 5 divided by 0 plus 3?",
    ]:
        items.append(_item(text, "calculator_error", ["calculator"], "hard"))

    # ---------------- system info ----------------
    for text in [
        "Show my current CPU usage.",
        "How much RAM is available on this machine?",
        "What is my disk usage?",
        "Show system information.",
        "How long has this system been running?",
        "Give me a quick health summary of this computer.",
        "Show CPU, RAM and disk details.",
        "What is the network status of this machine?",
        "How many CPU cores does this machine have?",
        "Is this machine low on memory?",
    ]:
        items.append(_item(text, "system_info", ["get_system_info"], "easy"))

    # ---------------- weather ----------------
    for city in CITIES:
        items.append(_item(f"What is the weather in {city}?", "weather", ["get_weather"], "easy"))
    for city in rng.sample(CITIES, 10):
        items.append(_item(f"Do I need an umbrella in {city} today?", "weather", ["get_weather"], "medium"))
    for city in BAD_CITIES:
        items.append(_item(f"What is the weather in {city}?", "weather_error", ["get_weather"], "hard"))

    # ---------------- wikipedia ----------------
    for topic in TOPICS:
        items.append(_item(f"Find information about {topic}.", "wikipedia", ["wikipedia_lookup"], "easy"))
    for topic in rng.sample(TOPICS, 10):
        items.append(_item(f"Look up {topic} and explain it in simple words.", "wikipedia", ["wikipedia_lookup"], "medium"))
    for topic in BAD_TOPICS:
        items.append(_item(f"Find information about {topic}.", "wikipedia_error", ["wikipedia_lookup"], "hard"))

    # ---------------- flaky tool ----------------
    for _ in range(10):
        items.append(_item("Run flaky_tool in success mode.", "flaky_success", ["flaky_tool"], "easy"))
    for secs in [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]:
        items.append(_item(
            f"Run flaky_tool in delay mode with a {secs}-second delay.",
            "flaky_delay", ["flaky_tool"], "medium"))
    for msg in ["Database connection lost", "Disk is full", "Permission denied",
                "Service unavailable", "Upstream API returned 500"]:
        items.append(_item(
            f"Run flaky_tool in error mode with the message '{msg}'.",
            "flaky_error", ["flaky_tool"], "medium"))
    for secs in [1, 1, 2, 2, 3, 3, 2, 1, 3, 2]:
        items.append(_item(
            f"Run flaky_tool in timeout mode with a {secs}-second timeout.",
            "flaky_timeout", ["flaky_tool"], "medium"))
    for _ in range(15):
        items.append(_item(
            rng.choice([
                "Test flaky_tool with intermittent failures.",
                "Run flaky_tool in intermittent mode.",
                "Check how reliable flaky_tool is in intermittent mode.",
            ]),
            "flaky_intermittent", ["flaky_tool"], "medium"))

    # ---------------- multi-tool ----------------
    for _ in range(20):
        a, b, city = rng.randint(10, 300), rng.randint(2, 50), rng.choice(CITIES)
        items.append(_item(
            f"Calculate {a} x {b} and tell me the weather in {city}.",
            "multi_tool", ["calculator", "get_weather"], "medium"))
    for _ in range(15):
        topic, city = rng.choice(TOPICS), rng.choice(CITIES)
        items.append(_item(
            f"Find information about {topic} and check the weather in {city}.",
            "multi_tool", ["wikipedia_lookup", "get_weather"], "medium"))
    for _ in range(10):
        a = rng.randint(10, 400)
        items.append(_item(
            f"Show my CPU usage and calculate {a} x 12.",
            "multi_tool", ["get_system_info", "calculator"], "medium"))
    for _ in range(15):
        c1, c2 = rng.sample(CITIES, 2)
        items.append(_item(
            f"Compare the weather in {c1} and {c2}.",
            "multi_tool", ["get_weather", "get_weather"], "medium"))
    for _ in range(10):
        topic, city, a = rng.choice(TOPICS), rng.choice(CITIES), rng.randint(5, 200)
        items.append(_item(
            f"Find information about {topic}, check the weather in {city}, and calculate {a} x {a}.",
            "multi_tool", ["wikipedia_lookup", "get_weather", "calculator"], "hard"))
    for _ in range(5):
        items.append(_item(
            "Show my RAM usage, then run flaky_tool in intermittent mode.",
            "multi_tool", ["get_system_info", "flaky_tool"], "hard"))

    # ---------------- no tool needed ----------------
    for text in GENERAL_QUESTIONS:
        items.append(_item(text, "no_tool", [], "easy"))

    # Remove exact duplicates, except flaky prompts: repeating them is
    # intentional (their latency and failures are random).
    result = []
    seen = set()
    for item in items:
        if item["category"].startswith("flaky"):
            result.append(item)
        elif item["prompt"] not in seen:
            seen.add(item["prompt"])
            result.append(item)

    rng.shuffle(result)
    return result


# =========================================================
# STYLE VARIANTS (more data: same task, different wording)
# =========================================================
# Wording changes prompt length and answer length, which changes
# token counts and latency. That gives the ML model useful variation.

STYLES = {
    "polite": lambda p: f"Hi, could you help me with something? {p} Thank you!",
    "verbose": lambda p: (
        "I am working on a project and I need some help with a small task. "
        f"{p} Please give a clear and detailed answer."
    ),
    "urgent": lambda p: f"URGENT: {p} Answer quickly.",
    "hinglish": lambda p: f"Bhai ek kaam kar do: {p}",
    "explain": lambda p: f"{p} Also explain how you got the answer, step by step.",
}


def build_tool_prompts(seed: int = 42, target: int = 1000) -> list:
    """Base prompts plus style variants, up to `target` prompts in total."""

    rng = random.Random(seed)
    base = _build_base_prompts(seed)

    if target <= len(base):
        return [
            {**item, "core_prompt": item["prompt"]}
            for item in base[:target]
        ]

    variants = []
    for item in base:
        for style, wrap in STYLES.items():
            variants.append({
                **item,
                "prompt": wrap(item["prompt"]),
                "core_prompt": item["prompt"],
                "style": style,
            })

    rng.shuffle(variants)

    result = [{**item, "core_prompt": item["prompt"]} for item in base]
    seen = {item["prompt"] for item in base if not item["category"].startswith("flaky")}

    for variant in variants:
        if len(result) >= target:
            break
        if variant["prompt"] in seen:
            continue
        seen.add(variant["prompt"])
        result.append(variant)

    rng.shuffle(result)
    return result