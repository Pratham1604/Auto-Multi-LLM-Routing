import json
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Define URLs of LLM providers
llm_urls = {
    "OpenAI": "https://platform.openai.com/docs/models",
    "Anthropic": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "Google": "https://ai.google.dev/gemini-api/docs/models",
    "Mistral": "https://docs.mistral.ai/getting-started/models/models_overview/",
    "Meta": "https://www.llama.com/docs/model-cards-and-prompt-formats/",
    "Cohere": "https://docs.cohere.com/docs/models",
    "Reka": "https://www.reka.ai/ourmodels",
    "01.ai": "https://huggingface.co/01-ai",
    "xAI (Grok)": "https://x.ai/api",
}


def get_rendered_html(url, wait_time=7):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    time.sleep(wait_time)  # Allow JS to render content
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    return soup


def extract_info(soup):
    text = soup.get_text(separator="\n", strip=True)
    lines = text.split("\n")
    extracted = {
        "Model Names": set(),
        "Pricing/Cost": set(),
        "Token Limit": set(),
        "Accuracy/Benchmarks": set(),
        "Capabilities": set(),
        "Release Info": set(),
        "Use Cases": set(),
        "Input/Output": set(),
        "Latency": set(),
        "API/Endpoint": set()
    }

    for line in lines:
        lower = line.lower()
        if "model" in lower and any(name in lower for name in ["claude", "gpt", "gemini", "mistral", "llama", "command-r", "reka", "mpt"]):
            extracted["Model Names"].add(line)
        if any(k in lower for k in ["price", "cost", "$", "pricing"]):
            extracted["Pricing/Cost"].add(line)
        if "token" in lower or "context length" in lower:
            extracted["Token Limit"].add(line)
        if any(k in lower for k in ["benchmark", "accuracy", "eval", "winogrande", "mmlu", "hellaswag"]):
            extracted["Accuracy/Benchmarks"].add(line)
        if any(k in lower for k in ["text", "image", "code", "multimodal", "audio", "video"]):
            extracted["Capabilities"].add(line)
        if any(k in lower for k in ["release", "version", "date"]):
            extracted["Release Info"].add(line)
        if any(k in lower for k in ["use case", "scenario", "application", "task"]):
            extracted["Use Cases"].add(line)
        if any(k in lower for k in ["input", "output", "tokens per second", "i/o", "rate"]):
            extracted["Input/Output"].add(line)
        if "latency" in lower or "response time" in lower:
            extracted["Latency"].add(line)
        if "endpoint" in lower or "api" in lower:
            extracted["API/Endpoint"].add(line)

    # Convert all sets to lists before returning
    return {key: list(value) for key, value in extracted.items()}



# Scrape and aggregate information
all_detailed_data = []

for provider, url in llm_urls.items():
    print(f"📡 Scraping {provider}...")
    try:
        soup = get_rendered_html(url)
        model_info = extract_info(soup)
        all_detailed_data.append({
            "Provider": provider,
            "Source URL": url,
            **model_info
        })
    except Exception as e:
        print(f"❌ Failed to scrape {provider}: {e}")

# ✅ Save JSON
with open("llm_models_detailed_info.json", "w", encoding="utf-8") as jf:
    json.dump(all_detailed_data, jf, indent=2, ensure_ascii=False)

# # ✅ Save CSV
# csv_fields = list(all_detailed_data[0].keys())
# with open("llm_models_detailed_info.csv", "w", encoding="utf-8", newline='') as cf:
#     writer = csv.DictWriter(cf, fieldnames=csv_fields)
#     writer.writeheader()
#     for row in all_detailed_data:
#         writer.writerow({
#             key: "\n".join(val) if isinstance(val, list) else val
#             for key, val in row.items()
#         })

# print("✅ Scraping complete. Data saved to 'llm_models_detailed_info.json' and 'llm_models_detailed_info.csv'")
