import requests
import json
import sqlite3
import time  # For rate limiting
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# Configurable parameters
NUM_BATCHES = 20  # Set to generate 200 recipes (20 batches * 10 recipes)
RECIPES_PER_BATCH = 10
MIN_IMAGE_WIDTH = 800
MIN_IMAGE_HEIGHT = 600

# Function to call Groq API for text generation
def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-70b-8192"
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        if result.get("choices"):
            return result["choices"][0]["message"]["content"].strip()
    raise Exception(f"Groq API error: {response.text}")

# Generate 100 batches of sensible ingredients
def generate_ingredient_batches(num_batches):
    prompt = f"Generate {num_batches} diverse lists of 5-10 sensible ingredients that could be found in a fridge or pantry and make sense together for cooking (e.g., for a meal theme like Italian, breakfast, or Asian). Each list should be a JSON array of strings. Output as a single JSON array of these lists. No extra text."
    response = call_groq(prompt)
    print("--- RAW GROQ RESPONSE (INGREDIENTS) ---")
    print(response)
    print("---------------------------------------")
    try:
        # Clean the response by removing markdown and extra whitespace
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise Exception("Failed to parse ingredient batches from LLM response.")

# Generate 10 recipes for a given batch of ingredients
def generate_recipes(ingredients, num_recipes):
    ingredients_str = json.dumps(ingredients)
    prompt = f"""Create {num_recipes} diverse, original recipes using primarily these ingredients: {ingredients_str}.
Each recipe should be well-described for a home cook.

Follow this exact JSON format for each recipe object:
{{
  "title": "A creative and descriptive title",
  "ingredients": [
    "string with measurement and ingredient, e.g., '1 cup of all-purpose flour'"
  ],
  "instructions": [
    "A clear, single step of the cooking process."
  ]
}}

Ensure all recipes have the same structure. Output as a single JSON array of these recipe objects. No extra text."""
    response = call_groq(prompt)
    try:
        # Clean the response by removing markdown and extra whitespace
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise Exception("Failed to parse recipes from LLM response.")

# Search multiple APIs for the best image (by resolution)
def find_best_image(query):
    # Refine query for better relevance
    search_query = f'"{query}" food photography'
    apis = [
        {"name": "Pexels", "url": f"https://api.pexels.com/v1/search?query={search_query}&per_page=5", "headers": {"Authorization": PEXELS_API_KEY}, "key": "photos", "src": lambda p: p["src"]["large"], "width": lambda p: p["width"], "height": lambda p: p["height"]},
        {"name": "Pixabay", "url": f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={search_query}&image_type=photo&per_page=5", "headers": {}, "key": "hits", "src": lambda p: p["largeImageURL"], "width": lambda p: p["imageWidth"], "height": lambda p: p["imageHeight"]},
        {"name": "Flickr", "url": f"https://www.flickr.com/services/feeds/photos_public.gne?tags={search_query},food&format=json&nojsoncallback=1", "headers": {}, "key": "items", "src": lambda p: p["media"]["m"].replace("_m", ""), "width": lambda _: MIN_IMAGE_WIDTH, "height": lambda _: MIN_IMAGE_HEIGHT}  # Flickr approx
    ]
    
    for api in apis:
        if not api["url"]: continue  # Skip if key missing
        try:
            response = requests.get(api["url"], headers=api["headers"])
            if response.status_code == 200:
                data = response.json()
                images = data.get(api["key"], [])
                for img in images:
                    width = api["width"](img)
                    height = api["height"](img)
                    if width >= MIN_IMAGE_WIDTH and height >= MIN_IMAGE_HEIGHT:
                        return api["src"](img)
        except Exception as e:
            print(f"API {api['name']} failed: {e}")
    return None  # No image found

# Set up SQLite DB
def setup_db():
    conn = sqlite3.connect('recipes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            title TEXT,
            ingredients_json TEXT,
            instructions_json TEXT,
            image_url TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

# Insert recipe into DB
def insert_recipe(cursor, batch_id, recipe, image_url):
    cursor.execute('''
        INSERT INTO recipes (batch_id, title, ingredients_json, instructions_json, image_url)
        VALUES (?, ?, ?, ?, ?)
    ''', (batch_id, recipe['title'], json.dumps(recipe['ingredients']), json.dumps(recipe['instructions']), image_url))

# Main function
def main():
    conn, cursor = setup_db()
    
    # Clear existing recipes before generating new ones
    print("Clearing existing recipes from the database...")
    cursor.execute("DELETE FROM recipes")
    conn.commit()
    
    try:
        batches = generate_ingredient_batches(NUM_BATCHES)
    except Exception as e:
        print(f"Fatal: Could not generate any ingredient batches. Error: {e}")
        return  # Exit if we can't get any batches

    total_recipes = 0
    
    for batch_id, ingredients in enumerate(batches, start=1):
        print(f"Processing batch {batch_id}/{NUM_BATCHES} with ingredients: {ingredients}")
        try:
            recipes = generate_recipes(ingredients, RECIPES_PER_BATCH)
        except Exception as e:
            print(f"  Could not generate recipes for this batch. Error: {e}")
            continue  # Skip to the next batch
        
        for recipe in recipes:
            try:
                query = f"{recipe['title']} food recipe"
            except KeyError:
                print(f"  Skipping malformed recipe (missing title): {recipe}")
                continue

            image_url = find_best_image(query)
            
            # Use a placeholder if no image is found
            if not image_url:
                image_url = "https://via.placeholder.com/800x600.png?text=No+Image+Found"

            insert_recipe(cursor, batch_id, recipe, image_url)
            total_recipes += 1
            print(f"  ({total_recipes}/{NUM_BATCHES * RECIPES_PER_BATCH}) Added recipe: {recipe['title']}")
            time.sleep(1)  # Rate limit pause
        
        conn.commit()  # Commit after each batch

        if batch_id < NUM_BATCHES:
            print("\nPausing for 10 seconds to respect API rate limits...\n")
            time.sleep(10)
        
    conn.close()
    print(f"Done! Populated database with {total_recipes} recipes across {NUM_BATCHES} batches.")

if __name__ == "__main__":
    main()
