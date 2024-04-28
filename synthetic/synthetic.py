import requests
import time
from bs4 import BeautifulSoup
from gemini import Gemini, GeminiHyperparameters
import json
import csv
import random
import time
import os

base_url = "https://en.wikinews.org/wiki/Special:Random"
visited_urls = set()
consecutive_visited_count = 0
max_consecutive_visited = 5

# Set the number of examples to include in the prompt
NUM_EXAMPLES_IN_PROMPT = 100

# Initialize the Gemini model
model = Gemini(model_name="gemini-pro")

def load_existing_source_urls():
    existing_urls = set()
    if os.path.exists("synthetic/wikinews_synthetic_data.jsonl"):
        with open("synthetic/wikinews_synthetic_data.jsonl", "r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                try:
                    data = json.loads(line)
                    if 'source_url' in data:
                        existing_urls.add(data['source_url'])
                except json.JSONDecodeError:
                    pass

    return existing_urls

# Load the NYT dataset
def load_nyt_dataset():
    nyt_data = []
    with open("data/NYT_dataset.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) < 7 or not row[2].strip() or not row[4].strip() or not row[6].strip():
                continue  # Skip rows with missing data or empty fields
            nyt_data.append({
                'title': row[2],
                'abstract': row[4],
                'keywords': eval(row[6]),
                'topic': row[3]
            })
    return nyt_data

# Function to extract the title from the Wikinews page
def extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    title_element = soup.find("span", class_="mw-page-title-main")
    if title_element:
        return title_element.get_text().strip()
    return ""

# Function to extract the content from the Wikinews page
def extract_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", class_="mw-content-ltr mw-parser-output")
    content = ""
    if content_div:
        for element in content_div.children:
            if element.name == "h2":
                break
            if element.name == "p":
                content += " " + element.get_text()
    return content.strip()

# Function to generate synthetic data using Gemini Pro
def generate_synthetic_data(title: str, content: str, examples: list[dict]) -> str:
    examples_json = "\n".join([f"```json\n{json.dumps(example, indent=2)}\n```" for example in examples])
    
    prompt = f"""
Title: {title}
Content: {content}

Generate JSON following the examples based on the given title and content. Provide the following fields in a JSON object:

abstract: A brief summary of the article's content, which rarely exceed a sentnece or two.
keywords: A list of relevant keywords for the article, which always includes at least one. Foreign, U.S., and Politics are NEVER allowed as keywords.
title: The title of the synthetic article.
topic: The topic of the article, which should be one of the following: Foreign, U.S., or Politics.

NEVER use new line characters (\n) in your response.

Use the following examples as a reference:

{examples_json}
Generated Synthetic JSON:
"""

    hyperparameters = GeminiHyperparameters(
        candidate_count=1,
        max_output_tokens=2048,
        temperature=1,
    )

    try:
        response = model.generate(prompt, hyperparameters=hyperparameters)
        print("_"*100)
        print(response)
        print("_"*100)
        # Remove new line characters from the response
        response = response.replace("\n", " ").strip()
        return response
    except Exception as e:
        print(f"An error occurred while generating synthetic data: {e}")
        # Log the error or perform any necessary error handling
        with open("synthetic/error_log.log", "a") as log_file:
            log_file.write(f"Error: {str(e)}\n")
            log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("-" * 50 + "\n")
        return ""

def parse_synthetic_data(synthetic_data: str) -> dict | None:
    # Check if the data is enclosed in a markdown code block and extract the JSON part
    if synthetic_data.startswith("```json") and synthetic_data.endswith("```"):
        # Remove the markdown code block syntax to isolate the JSON string
        json_data = synthetic_data[7:-3].strip()
    else:
        # If no markdown code block is found, consider the entire string as JSON data
        json_data = synthetic_data

    # Remove any new line characters from the JSON data
    json_data = json_data.replace("\n", " ")

    try:
        synthetic_dict = json.loads(json_data)
        if isinstance(synthetic_dict, dict):
            return synthetic_dict
    except json.JSONDecodeError:
        pass

    return None

# Function to check if the synthetic data meets the specification
def is_valid_synthetic_data(synthetic_dict: dict) -> bool:
    required_keys = ['title', 'abstract', 'keywords', 'topic']
    if synthetic_dict and all(key in synthetic_dict for key in required_keys):
        keywords = synthetic_dict['keywords']
        # Ensure there is at least one keyword
        if len(keywords) > 0:
            if synthetic_dict['topic'] in ['Foreign', 'U.S.', 'Politics']:
                # Check if any of the keywords match the topic
                if not any(keyword.lower() in ['u.s.', 'foreign', 'politics'] for keyword in keywords):
                    return True
    return False

# Function to retrieve and process a random Wikinews page
def process_wikinews_page(nyt_data):
    global consecutive_visited_count, visited_urls

    response = requests.get(base_url)
    if response.status_code == 200:
        current_url = response.url
        if "/Crosswords/" in current_url:
            print(f"Skipping crossword URL: {current_url}")
            return current_url, None
        if "/Wikinews_Shorts:" in current_url:
            print(f"Skipping Wikinews Shorts URL: {current_url}")
            return current_url, None
        if current_url in visited_urls:
            consecutive_visited_count += 1
            print(f"Skipping duplicate URL: {current_url}")
            time.sleep(1)
            if consecutive_visited_count >= max_consecutive_visited:
                print("Giving up after visiting previously visited pages 5 times in a row.")
                return current_url, None
            return current_url, None
        else:
            consecutive_visited_count = 0
            visited_urls.add(current_url)
            print(f"New URL: {current_url}")

            title = extract_title(response.text)
            content = extract_content(response.text)

            examples = random.sample(nyt_data, min(NUM_EXAMPLES_IN_PROMPT, len(nyt_data)))

            max_tries = 10
            for _ in range(max_tries):
                synthetic_data = generate_synthetic_data(title, content, examples)
                synthetic_dict = parse_synthetic_data(synthetic_data)

                if is_valid_synthetic_data(synthetic_dict):
                    synthetic_dict['source_url'] = current_url  # Add the "source_url" field
                    return current_url, synthetic_dict

            print(f"Failed to generate valid synthetic data after {max_tries} tries. Skipping this article.")
            return current_url, None
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None, None

# Function to start the synthetic data generation process
def generate_synthetic_data_jsonl(max_iterations):
    global visited_urls

    nyt_data = load_nyt_dataset()
    visited_urls = load_existing_source_urls()

    with open("synthetic/wikinews_synthetic_data.jsonl", "a", encoding="utf-8") as jsonl_file:
        iteration = 0
        while iteration < max_iterations:
            try:
                source_url, synthetic_dict = process_wikinews_page(nyt_data)
                if source_url is not None and source_url not in visited_urls:
                    if synthetic_dict is None:
                        max_tries = 10
                        for _ in range(max_tries):
                            synthetic_data = generate_synthetic_data(extract_title(response.text), extract_content(response.text), random.sample(nyt_data, min(NUM_EXAMPLES_IN_PROMPT, len(nyt_data))))
                            synthetic_dict = parse_synthetic_data(synthetic_data)

                            if is_valid_synthetic_data(synthetic_dict):
                                synthetic_dict['source_url'] = source_url
                                break
                        else:
                            print(f"Failed to generate valid synthetic data after {max_tries} tries. Skipping this article.")
                            continue

                    try:
                        synthetic_dict['model'] = 'gemini-pro'  # Add the "model" field
                        jsonl_file.write(json.dumps(synthetic_dict, ensure_ascii=False) + "\n")
                        jsonl_file.flush()  # Flush the buffer to ensure data is written to the file
                        print(f"Generated Synthetic Data: {synthetic_dict}")
                        iteration += 1
                        visited_urls.add(source_url)
                    except Exception as e:
                        print(f"An error occurred while writing to the JSONL file: {e}")
                        # Log the error or perform any necessary error handling
                        with open("error_log.log", "a") as log_file:
                            log_file.write(f"Error: {str(e)}\n")
                            log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            log_file.write("-" * 50 + "\n")
                else:
                    print(f"Skipping duplicate source URL: {source_url}")

                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                # Log the error or perform any necessary error handling
                with open("error_log.log", "a") as log_file:
                    log_file.write(f"Error: {str(e)}\n")
                    log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write("-" * 50 + "\n")

# Run the synthetic data generation process
generate_synthetic_data_jsonl(max_iterations=25000)
