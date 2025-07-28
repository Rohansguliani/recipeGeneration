import requests
import json
import sqlite3
import time
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY") # Re-adding Pixabay
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Goal-oriented parameters
TARGET_RECIPE_COUNT = 300
MINIMUM_IMAGE_SCORE = 78      # The quality gate for images

# Generation parameters
RECIPES_PER_THEME = 10
CANDIDATE_IMAGE_COUNT = 7

# Image constraints
MIN_IMAGE_WIDTH = 800
MIN_IMAGE_HEIGHT = 600

# --- LLM and API Functions ---

def call_groq(prompt, model="llama3-70b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": prompt}], "model": model}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        if result.get("choices"):
            return result["choices"][0]["message"]["content"].strip()
    raise Exception(f"Groq API error: {response.text}")

def generate_recipe_themes(count, existing_themes):
    prompt = f"""Generate {count} simple and practical recipe themes.
The themes should be common and based on meal types, cuisines, or key ingredients.
Examples: 'Quick Weeknight Dinners', 'Classic Italian Pasta', 'Healthy Chicken Dishes', 'Vegetarian Breakfast Ideas', 'Simple Seafood Recipes'.
Ensure the themes are unique and not similar to these existing ones: {list(existing_themes)}.
Output as a single JSON array of strings. No extra text."""
    response = call_groq(prompt, model="llama3-8b-8192") # Use a faster model for simple generation
    try:
        return json.loads(response.strip().replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        print("Warning: Failed to parse recipe themes from LLM.")
        return [f"Random Batch {int(time.time())}"]

def generate_recipes_for_theme(theme, num_recipes):
    prompt = f"""Create {num_recipes} diverse, original recipes based on the theme: '{theme}'.
Each recipe must be a real, valid recipe, well-described for a home cook.

Follow this exact JSON format for each recipe object:
{{
  "title": "A creative and descriptive title",
  "description": "A short, enticing one-sentence description of the dish.",
  "cuisine": "A single, common cuisine type (e.g., 'Italian', 'Mexican', 'Indian', 'Japanese').",
  "difficulty": "Either 'Easy', 'Medium', or 'Hard'.",
  "ingredients": ["e.g., '1 cup of all-purpose flour, sifted'"],
  "instructions": ["A clear, single step of the cooking process. Be descriptive."]
}}
Output as a single JSON array. No extra text."""
    response = call_groq(prompt)
    try:
        return json.loads(response.strip().replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse recipes for theme '{theme}'.")
        return []

GEMINI_IS_RATE_LIMITED = False
GEMINI_COOLDOWN_UNTIL = 0

def rate_image_with_gemini(recipe_title, image_url):
    global GEMINI_IS_RATE_LIMITED, GEMINI_COOLDOWN_UNTIL

    if GEMINI_IS_RATE_LIMITED and time.time() < GEMINI_COOLDOWN_UNTIL:
        # We are still in the cooldown period, don't make an API call
        return 0
    
    # If the cooldown has passed, reset the flag
    if GEMINI_IS_RATE_LIMITED:
        GEMINI_IS_RATE_LIMITED = False

    if not GOOGLE_API_KEY:
        return 0 

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Download the image to send to the API
        image_response = requests.get(image_url, stream=True)
        if image_response.status_code != 200:
            print(f"    - Gemini: Failed to download image {image_url}")
            return 0
        
        image_parts = [{"mime_type": "image/jpeg", "data": image_response.content}]
        prompt_parts = [
            image_parts[0],
            f"\nRate how well this image matches the recipe title '{recipe_title}' on a scale from 0 to 10. Consider accuracy and visual appeal. Return ONLY the integer score.",
        ]

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_parts)
        
        score = int(response.text.strip())
        print(f"    - Gemini Likeness Score: {score}/10")
        return score
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str and "quota" in error_str:
            print("\n    !!!! Gemini daily rate limit reached. Disabling AI scoring for 1 hour. !!!!\n")
            GEMINI_IS_RATE_LIMITED = True
            GEMINI_COOLDOWN_UNTIL = time.time() + 3600
            return 0 # Return neutral score

        print(f"    - Gemini: Could not rate image. Error: {e}")
        return 0 


# --- Image Searching & Scoring Engine ---

DISH_TYPES = [
    "soup", "stew", "curry", "chili", "salad", "pasta", "risotto", "paella",
    "stir-fry", "fried rice", "tacos", "burritos", "quesadilla", "enchiladas",
    "pizza", "casserole", "frittata", "omelette", "quiche", "kabobs", "skewers",
    "sandwich", "burger", "wrap", "dumplings", "noodles", "meatballs", "sushi",
    "roast", "grilled", "baked", "fried", "bites", "wings", "cake"
]

def score_image(recipe_title, image_metadata, dish_type, image_url):
    # 1. Heuristic Score (fast, keyword-based)
    heuristic_score = 0
    title_lower = recipe_title.lower()
    title_words = set(re.findall(r'\w+', title_lower))
    metadata_str = str(image_metadata).lower()

    if dish_type and dish_type in metadata_str:
        heuristic_score += 50
        if metadata_str.find(dish_type) < 50:
            heuristic_score += 15
    
    primary_ingredients = set(title_lower.replace(dish_type, "").split()) if dish_type else title_words
    for ingredient in primary_ingredients:
        if ingredient in metadata_str:
            heuristic_score += 10
    
    metadata_words = set(re.findall(r'\w+', metadata_str))
    heuristic_score += len(title_words.intersection(metadata_words)) * 2

    if dish_type == "cake" and any(w in title_lower for w in ["chicken", "beef", "fish", "savory"]):
        if "dessert" in metadata_str or "sweet" in metadata_str:
            heuristic_score -= 100
    
    if 'food' in metadata_words and len(metadata_words) < 5:
        heuristic_score -= 10

    # 2. AI Vision Score (slower, more accurate)
    # The AI's 0-10 score is weighted heavily
    ai_score = rate_image_with_gemini(recipe_title, image_url) * 10 
    
    # Return the combined score
    return heuristic_score + ai_score

UNSPLASH_REQUESTS_COUNT = 0
UNSPLASH_RATE_LIMIT = 50
LAST_UNSPLASH_REQUEST_TIME = time.time()

def can_use_unsplash():
    global UNSPLASH_REQUESTS_COUNT, LAST_UNSPLASH_REQUEST_TIME
    if time.time() - LAST_UNSPLASH_REQUEST_TIME > 3600:
        UNSPLASH_REQUESTS_COUNT = 0
        LAST_UNSPLASH_REQUEST_TIME = time.time()
    return UNSPLASH_REQUESTS_COUNT < UNSPLASH_RATE_LIMIT

def find_best_image(title, used_urls):
    print(f"\n--- Finding image for: {title} ---")
    dish_type = next((dish for dish in DISH_TYPES if dish in title.lower()), None)
    search_query = f'food photography "{title}"'

    # --- API Provider Fallback Loop ---
    api_providers = ["Unsplash", "Pexels", "Pixabay"]

    for provider in api_providers:
        print(f"  Trying provider: {provider}...")
        candidates = []

        # 1. Gather candidates from the current provider
        if provider == "Unsplash" and UNSPLASH_ACCESS_KEY and can_use_unsplash():
            global UNSPLASH_REQUESTS_COUNT
            url = f"https://api.unsplash.com/search/photos?query={search_query}&per_page={CANDIDATE_IMAGE_COUNT}&orientation=landscape"
            headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
            try:
                response = requests.get(url, headers=headers)
                UNSPLASH_REQUESTS_COUNT += 1
                if response.status_code == 200:
                    for img in response.json().get("results", []):
                        if img["width"] >= MIN_IMAGE_WIDTH:
                            candidates.append({"url": img["urls"]["regular"], "metadata": f"{img.get('description', '')} {img.get('alt_description', '')}", "source": "Unsplash"})
            except Exception as e:
                print(f"    Warning: Unsplash request failed: {e}")

        elif provider == "Pexels" and PEXELS_API_KEY:
            url = f"https://api.pexels.com/v1/search?query={search_query}&per_page={CANDIDATE_IMAGE_COUNT}"
            headers = {"Authorization": PEXELS_API_KEY}
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    for img in response.json().get("photos", []):
                        if img["width"] >= MIN_IMAGE_WIDTH:
                            candidates.append({"url": img["src"]["large2x"], "metadata": img.get('alt', ''), "source": "Pexels"})
            except Exception as e:
                print(f"    Warning: Pexels request failed: {e}")
        
        elif provider == "Pixabay" and PIXABAY_API_KEY:
            url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={search_query}&per_page={CANDIDATE_IMAGE_COUNT}&image_type=photo"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    for img in response.json().get("hits", []):
                        if img["imageWidth"] >= MIN_IMAGE_WIDTH:
                            candidates.append({"url": img["largeImageURL"], "metadata": img.get('tags', ''), "source": "Pixabay"})
            except Exception as e:
                print(f"    Warning: Pixabay request failed: {e}")

        # 2. Score candidates and check against quality gate
        if candidates:
            scored_candidates = []
            for cand in candidates:
                # Pass the image URL to the scoring function now
                score = score_image(title, cand["metadata"], dish_type, cand["url"])
                scored_candidates.append({"score": score, "candidate": cand})

            sorted_candidates = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

            # Find the best unique image from this provider's results
            for sc in sorted_candidates:
                if sc['candidate']['url'] not in used_urls:
                    # Quality Gate Check
                    if sc['score'] >= MINIMUM_IMAGE_SCORE:
                        print(f"    SUCCESS! Found high-quality image on {provider} with score {sc['score']}.")
                        print(f"    --> {sc['candidate']['url']}")
                        return sc['candidate']['url'], sc['score'], dish_type
                    else:
                        # The best available image from this provider is not good enough
                        print(f"    Provider {provider} did not meet quality gate. Best score was {sc['score']}.")
                        break # Stop checking this provider's images and move to the next provider
        else:
            print(f"    No results from {provider}.")
    
    # If the loop finishes, no suitable image was found
    print(f"  --> All providers failed to find a unique, high-quality image for '{title}'.")
    return None, 0, dish_type

# --- Database Functions ---

def setup_db():
    conn = sqlite3.connect('recipes.db')
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    cursor = conn.cursor()
    # cursor.execute('DROP TABLE IF EXISTS recipes') # Removed to make script resumable
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            cuisine TEXT,
            difficulty TEXT,
            dish_type TEXT,
            ingredients_json TEXT,
            instructions_json TEXT,
            image_url TEXT,
            image_score INTEGER,
            theme TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def insert_recipe(cursor, recipe_data):
    cursor.execute('''
        INSERT INTO recipes (title, description, cuisine, difficulty, dish_type, ingredients_json, instructions_json, image_url, image_score, theme)
        VALUES (:title, :description, :cuisine, :difficulty, :dish_type, :ingredients_json, :instructions_json, :image_url, :image_score, :theme)
    ''', recipe_data)

# --- Main Control Loop ---

def main():
    conn, cursor = setup_db()
    
    # Load current state from the database to make the script resumable
    cursor.execute("SELECT COUNT(id) FROM recipes")
    current_recipe_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT DISTINCT theme FROM recipes")
    generated_themes = {row['theme'] for row in cursor.fetchall()}

    cursor.execute("SELECT image_url FROM recipes")
    used_image_urls = {row['image_url'] for row in cursor.fetchall()}
    
    if current_recipe_count > 0:
        print(f"--- Resuming recipe generation ---")
        print(f"Found {current_recipe_count} recipes and {len(used_image_urls)} used images in the database.")
    else:
        print(f"--- Starting Quality-Gated Recipe Generation ---")

    print(f"Goal: Collect {TARGET_RECIPE_COUNT} recipes with an image score of {MINIMUM_IMAGE_SCORE} or higher.")
    
    while current_recipe_count < TARGET_RECIPE_COUNT:
        print(f"\n--- Progress: {current_recipe_count} / {TARGET_RECIPE_COUNT} ---")
        
        # 1. Generate new, unique themes
        new_themes = generate_recipe_themes(5, generated_themes)
        if not new_themes:
            print("Could not generate new themes, waiting...")
            time.sleep(5)
            continue
        
        for theme in new_themes:
            if theme in generated_themes or current_recipe_count >= TARGET_RECIPE_COUNT:
                continue
            
            generated_themes.add(theme)
            print(f"\nGenerating recipes for theme: '{theme}'")
            recipes_for_theme = generate_recipes_for_theme(theme, RECIPES_PER_THEME)

            for recipe in recipes_for_theme:
                if current_recipe_count >= TARGET_RECIPE_COUNT:
                    break
                try:
                    title = recipe['title']
                except KeyError:
                    continue

                image_url, score, dish_type = find_best_image(title, used_image_urls)
                
                if image_url and score >= MINIMUM_IMAGE_SCORE:
                    print(f"  --> SUCCESS! Recipe '{title}' passed quality gate with score {score}.")
                    recipe_data = {
                        "title": title,
                        "description": recipe.get('description', ''),
                        "cuisine": recipe.get('cuisine', 'Unknown'),
                        "difficulty": recipe.get('difficulty', 'Unknown'),
                        "dish_type": dish_type,
                        "ingredients_json": json.dumps(recipe.get('ingredients', [])),
                        "instructions_json": json.dumps(recipe.get('instructions', [])),
                        "image_url": image_url,
                        "image_score": score,
                        "theme": theme
                    }
                    insert_recipe(cursor, recipe_data)
                    used_image_urls.add(image_url) # Add to our set of used URLs
                    conn.commit()
                    current_recipe_count += 1 # Increment our counter
                else:
                    print(f"  --> SKIPPED. Recipe '{title}' did not meet quality gate (Score: {score}) or no unique image was found.")
                
                time.sleep(2) # Increased sleep to be gentler on APIs
            
    conn.close()
    print(f"\n--- Generation Complete! ---")
    print(f"Successfully collected and stored {current_recipe_count} high-quality recipes in recipes.db.")

if __name__ == "__main__":
    main()
