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
SHUTTERSTOCK_TOKEN = os.getenv("SHUTTERSTOCK_TOKEN")

# Goal-oriented parameters
TARGET_RECIPE_COUNT = 10000  # Increased to 10,000
MINIMUM_IMAGE_SCORE = 85  # The quality gate for images
AUDIT_INTERVAL_HOURS = 2  # Run quality audit every 2 hours

# Generation parameters
RECIPES_PER_THEME = 10
CANDIDATE_IMAGE_COUNT = 7
# Ensure each provider returns between 5 and 10 candidates
CANDIDATES_PER_PROVIDER = max(5, min(10, CANDIDATE_IMAGE_COUNT))

# Image constraints
MIN_IMAGE_WIDTH = 800
MIN_IMAGE_HEIGHT = 600

# --- LLM and API Functions ---

def call_groq(prompt, model="llama3-70b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": prompt}], "model": model}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            # If the call is successful, parse and return the response
            if response.status_code == 200:
                result = response.json()
                if result.get("choices"):
                    return result["choices"][0]["message"]["content"].strip()
            
            # --- Start of new error handling logic ---
            response_text = response.text
            
            # Check specifically for the rate limit error
            if "rate_limit_exceeded" in response_text:
                print(f"    Groq rate limit hit (Attempt {attempt + 1}/{max_retries})...")
                
                # Use regex to find the cooldown time, e.g., "6m5.437s"
                match = re.search(r'try again in (?:(\d+)m)?(?:([\d.]+)s)?', response_text)
                
                if match:
                    minutes = float(match.group(1)) if match.group(1) else 0
                    seconds = float(match.group(2)) if match.group(2) else 0
                    cooldown = (minutes * 60) + seconds + 2 # Add a 2-second buffer
                    
                    print(f"    Cooldown detected: {minutes}m {seconds}s. Pausing script for {cooldown:.1f} seconds...")
                    time.sleep(cooldown)
                    print(f"    Cooldown finished. Retrying API call...")
                    continue # Retry the loop
            
            # Check for service unavailability errors
            if "service unavailable" in response_text.lower() or "internal_server_error" in response_text:
                wait_time = min(30 * (2 ** attempt), 300)  # Exponential backoff: 30s, 60s, 120s, 240s, 300s max
                print(f"    Groq service unavailable (Attempt {attempt + 1}/{max_retries}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            # For any other non-200 status, raise the error to be handled or logged
            raise Exception(f"Groq API error: {response_text}")
        
        except requests.exceptions.RequestException as e:
            wait_time = min(30 * (2 ** attempt), 300)  # Same exponential backoff for network errors
            print(f"    Network error calling Groq: {e}. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

    raise Exception(f"Groq API call failed after {max_retries} retries.")


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


def rate_image_with_gemini(recipe_title, image_url, dish_type=None):
    global GEMINI_IS_RATE_LIMITED, GEMINI_COOLDOWN_UNTIL

    if GEMINI_IS_RATE_LIMITED and time.time() < GEMINI_COOLDOWN_UNTIL:
        return 0

    if GEMINI_IS_RATE_LIMITED:
        GEMINI_IS_RATE_LIMITED = False

    if not GOOGLE_API_KEY:
        return 0

    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        image_response = requests.get(image_url, stream=True)
        if image_response.status_code != 200:
            print(f"    - Gemini: Failed to download image {image_url}")
            return 0

        image_parts = [{"mime_type": "image/jpeg", "data": image_response.content}]
        rules = (
            "You are a strict food photo judge. Rate how well this image matches the dish '"
            + recipe_title
            + "'.\n"
            "Rules:\n"
            "- Prefer a realistic, appetizing, plated, cooked dish.\n"
            "- Do NOT show product packaging (cans, jars, boxes), logos/labels, menus, or obvious stock mockups.\n"
            "- Do NOT show a photo of only packaged food or a can; that should be rated 0.\n"
            "- Avoid illustrations, clipart, or cartoons; rate 0 if detected.\n"
            "- Avoid purely raw ingredients unless the dish itself is a raw preparation.\n"
        )
        if dish_type == "soup":
            rules += "- If the dish is soup and the image shows a can of soup or packaged soup, rate 0.\n"
        rules += "Return ONLY an integer from 0 to 10."

        prompt_parts = [image_parts[0], rules]

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
            return 0

        print(f"    - Gemini: Could not rate image. Error: {e}")
        return 0


def audit_recipe_image_with_gemini(recipe_title, image_url):
    """Use Gemini to audit how well an existing image matches its recipe"""
    global GEMINI_IS_RATE_LIMITED, GEMINI_COOLDOWN_UNTIL

    if GEMINI_IS_RATE_LIMITED and time.time() < GEMINI_COOLDOWN_UNTIL:
        return 0

    if GEMINI_IS_RATE_LIMITED:
        GEMINI_IS_RATE_LIMITED = False

    if not GOOGLE_API_KEY:
        return 0

    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        image_response = requests.get(image_url, stream=True)
        if image_response.status_code != 200:
            return 0

        image_parts = [{"mime_type": "image/jpeg", "data": image_response.content}]
        dish_type = next((dish for dish in DISH_TYPES if dish in recipe_title.lower()), None)
        prompt_text = (
            "You are a culinary expert and food photographer. Analyze this image and rate how well it represents the recipe: '"
            + recipe_title
            + "'.\n"
            "Consider: accuracy, quality, relevance. Prefer a plated, cooked dish.\n"
            "Hard penalties (rate 0) for: product packaging (cans, jars, boxes), logos/labels, menus, illustrations, or obvious stock mockups.\n"
        )
        if dish_type == "soup":
            prompt_text += "If the dish is soup and the image shows a can of soup or packaged soup, rate 0.\n"
        prompt_text += "Return ONLY an integer from 0 to 10."

        prompt_parts = [image_parts[0], prompt_text]

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_parts)

        score = int(response.text.strip())
        return score
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str and "quota" in error_str:
            print("\n    !!!! Gemini daily rate limit reached during audit. Disabling AI scoring for 1 hour. !!!!\n")
            GEMINI_IS_RATE_LIMITED = True
            GEMINI_COOLDOWN_UNTIL = time.time() + 3600
            return 0

        return 0

# --- Image Searching & Scoring Engine ---

DISH_TYPES = [
    "soup", "stew", "curry", "chili", "salad", "pasta", "risotto", "paella",
    "stir-fry", "fried rice", "tacos", "burritos", "quesadilla", "enchiladas",
    "pizza", "casserole", "frittata", "omelette", "quiche", "kabobs", "skewers",
    "sandwich", "burger", "wrap", "dumplings", "noodles", "meatballs", "sushi",
    "roast", "grilled", "baked", "fried", "bites", "wings", "cake"
]

NEGATIVE_PACKAGING_KEYWORDS = {
    "can", "canned", "tin", "tinned", "jar", "boxed", "box", "packaging",
    "package", "label", "logo", "brand", "wrapper", "carton", "bottle", "can of"
}
NEGATIVE_NON_FOOD_VISUALS = {"illustration", "vector", "clipart", "cartoon", "icon", "graphic", "logo"}


def is_packaging_or_non_food(metadata_str: str, dish_type: str | None) -> bool:
    text = str(metadata_str).lower()
    has_packaging = any(k in text for k in NEGATIVE_PACKAGING_KEYWORDS)
    has_non_food = any(k in text for k in NEGATIVE_NON_FOOD_VISUALS)
    if dish_type == "soup" and ("can of soup" in text or ("soup" in text and "can" in text)):
        return True
    return has_packaging or has_non_food


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

    # Penalize product packaging or non-food depictions
    if is_packaging_or_non_food(metadata_str, dish_type):
        heuristic_score -= 100

    # 2. AI Vision Score (slower, more accurate)
    # The AI's 0-10 score is weighted heavily
    ai_score = rate_image_with_gemini(recipe_title, image_url, dish_type) * 10

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


def find_best_image(title, used_urls, allow_below_threshold: bool = False, min_score: int | None = None):
    print(f"\n--- Finding image for: {title} ---")
    dish_type = next((dish for dish in DISH_TYPES if dish in title.lower()), None)
    search_query = f'food photography "{title}"'

    all_candidates = []

    # Unsplash
    if UNSPLASH_ACCESS_KEY and can_use_unsplash():
        global UNSPLASH_REQUESTS_COUNT
        try:
            url = (
                f"https://api.unsplash.com/search/photos?query={search_query}"
                f"&per_page={CANDIDATES_PER_PROVIDER}&orientation=landscape"
            )
            headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
            response = requests.get(url, headers=headers)
            UNSPLASH_REQUESTS_COUNT += 1
            if response.status_code == 200:
                for img in response.json().get("results", []):
                    if img.get("width", 0) >= MIN_IMAGE_WIDTH:
                        metadata = f"{img.get('description', '')} {img.get('alt_description', '')}"
                        all_candidates.append({
                            "url": img["urls"]["regular"],
                            "metadata": metadata,
                            "source": "Unsplash",
                        })
        except Exception as e:
            print(f"    Warning: Unsplash request failed: {e}")

    # Pexels
    if PEXELS_API_KEY:
        try:
            url = f"https://api.pexels.com/v1/search?query={search_query}&per_page={CANDIDATES_PER_PROVIDER}"
            headers = {"Authorization": PEXELS_API_KEY}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                for img in response.json().get("photos", []):
                    if img.get("width", 0) >= MIN_IMAGE_WIDTH:
                        all_candidates.append({
                            "url": img["src"]["large2x"],
                            "metadata": img.get("alt", ""),
                            "source": "Pexels",
                        })
        except Exception as e:
            print(f"    Warning: Pexels request failed: {e}")

    # Pixabay
    if PIXABAY_API_KEY:
        try:
            url = (
                f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={search_query}"
                f"&per_page={CANDIDATES_PER_PROVIDER}&image_type=photo&orientation=horizontal"
            )
            response = requests.get(url)
            if response.status_code == 200:
                for img in response.json().get("hits", []):
                    if img.get("imageWidth", 0) >= MIN_IMAGE_WIDTH:
                        all_candidates.append({
                            "url": img["largeImageURL"],
                            "metadata": img.get("tags", ""),
                            "source": "Pixabay",
                        })
        except Exception as e:
            print(f"    Warning: Pixabay request failed: {e}")

    # Shutterstock
    if SHUTTERSTOCK_TOKEN:
        try:
            url = (
                f"https://api.shutterstock.com/v2/images/search?query={search_query}"
                f"&per_page={CANDIDATES_PER_PROVIDER}&orientation=horizontal"
            )
            headers = {"Authorization": f"Bearer {SHUTTERSTOCK_TOKEN}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                for img in response.json().get("data", []):
                    assets = img.get("assets", {})
                    chosen = None
                    for key in ("preview_1000", "preview", "huge_thumb"):
                        if key in assets and assets[key].get("url"):
                            chosen = assets[key]
                            break
                    if chosen and chosen.get("width", 0) >= MIN_IMAGE_WIDTH:
                        metadata = img.get("description", "")
                        if isinstance(img.get("keywords"), list):
                            metadata = f"{metadata} {' '.join(img['keywords'])}"
                        all_candidates.append({
                            "url": chosen.get("url"),
                            "metadata": metadata,
                            "source": "Shutterstock",
                        })
        except Exception as e:
            print(f"    Warning: Shutterstock request failed: {e}")

    if not all_candidates:
        print("    No results from any provider.")
        print(f"  --> All providers failed to find a unique, high-quality image for '{title}'.")
        return None, 0, dish_type

    scored_candidates = []
    for cand in all_candidates:
        if cand["url"] in used_urls:
            continue
        score = score_image(title, cand["metadata"], dish_type, cand["url"])
        scored_candidates.append({"score": score, "candidate": cand})

    if not scored_candidates:
        print("    All candidate URLs are already used or invalid.")
        return None, 0, dish_type

    sorted_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
    best = sorted_candidates[0]
    threshold = MINIMUM_IMAGE_SCORE if min_score is None else int(min_score)
    if best["score"] >= threshold:
        print(f"    SUCCESS! Best image from {best['candidate']['source']} scored {best['score']}.")
        print(f"    --> {best['candidate']['url']}")
        return best["candidate"]["url"], best["score"], dish_type

    if allow_below_threshold:
        print(f"    Using best available below threshold ({best['score']}). Source: {best['candidate']['source']}")
        print(f"    --> {best['candidate']['url']}")
        return best["candidate"]["url"], best["score"], dish_type

    print(f"    No candidate met the quality gate. Best score was {best['score']} from {best['candidate']['source']}.")
    return None, 0, dish_type

# --- Database Functions ---

def setup_db():
    conn = sqlite3.connect('recipes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
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
            theme TEXT,
            nutrition_json TEXT,
            vegan INTEGER DEFAULT 0,
            vegetarian INTEGER DEFAULT 0,
            gluten_free INTEGER DEFAULT 0,
            dairy_free INTEGER DEFAULT 0,
            nut_free INTEGER DEFAULT 0,
            keto_friendly INTEGER DEFAULT 0
        )
    ''')
    conn.commit()

    # Apply schema migrations for existing tables missing columns
    apply_schema_migrations(conn, cursor)

    return conn, cursor


def apply_schema_migrations(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(recipes)")
    columns = {row[1] for row in cursor.fetchall()}
    add_columns_sql = []
    if 'nutrition_json' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN nutrition_json TEXT")
    if 'vegan' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN vegan INTEGER DEFAULT 0")
    if 'vegetarian' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN vegetarian INTEGER DEFAULT 0")
    if 'gluten_free' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN gluten_free INTEGER DEFAULT 0")
    if 'dairy_free' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN dairy_free INTEGER DEFAULT 0")
    if 'nut_free' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN nut_free INTEGER DEFAULT 0")
    if 'keto_friendly' not in columns:
        add_columns_sql.append("ALTER TABLE recipes ADD COLUMN keto_friendly INTEGER DEFAULT 0")

    for sql in add_columns_sql:
        try:
            cursor.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            # Column may already exist due to race; ignore
            pass


def insert_recipe(cursor, recipe_data):
    cursor.execute('''
        INSERT INTO recipes (
            title, description, cuisine, difficulty, dish_type,
            ingredients_json, instructions_json, image_url, image_score, theme,
            nutrition_json, vegan, vegetarian, gluten_free, dairy_free, nut_free, keto_friendly
        )
        VALUES (
            :title, :description, :cuisine, :difficulty, :dish_type,
            :ingredients_json, :instructions_json, :image_url, :image_score, :theme,
            :nutrition_json, :vegan, :vegetarian, :gluten_free, :dairy_free, :nut_free, :keto_friendly
        )
    ''', recipe_data)


def estimate_nutrition(ingredients: list, title: str) -> dict:
    """Estimate nutrition per serving using LLM. Returns dict with common fields.
    Fallbacks to zeros on parse errors.
    """
    base = {
        "calories": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0,
        "fiber_g": 0,
        "sugar_g": 0,
        "saturated_fat_g": 0,
        "cholesterol_mg": 0,
        "sodium_mg": 0
    }

    try:
        prompt = (
            "Estimate nutrition for this recipe per serving. Return ONLY compact JSON with keys: "
            "calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, saturated_fat_g, cholesterol_mg, sodium_mg.\n"
            f"Title: {title}\nIngredients: {json.dumps(ingredients)}"
        )
        response = call_groq(prompt, model="llama3-8b-8192")
        text = response.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        # Merge to ensure all keys exist and are numeric
        for k in base.keys():
            v = data.get(k, 0)
            try:
                base[k] = float(v)
            except Exception:
                base[k] = 0
    except Exception:
        pass
    return base


def classify_dietary_flags(ingredients: list, title: str) -> dict:
    """Classify dietary booleans using LLM, with simple heuristics fallback."""
    default_flags = {
        "vegan": False,
        "vegetarian": False,
        "gluten_free": False,
        "dairy_free": False,
        "nut_free": False,
        "keto_friendly": False,
    }

    try:
        prompt = (
            "Given a recipe title and ingredients, answer ONLY JSON booleans for: "
            "vegan, vegetarian, gluten_free, dairy_free, nut_free, keto_friendly.\n"
            "Assume typical substitutions are not used unless stated.\n"
            f"Title: {title}\nIngredients: {json.dumps(ingredients)}"
        )
        text = call_groq(prompt, model="llama3-8b-8192").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        flags = {k: bool(data.get(k, False)) for k in default_flags.keys()}
        return flags
    except Exception:
        # Heuristic fallback
        ing_text = " ".join(map(str, ingredients)).lower()
        meats = ["beef", "pork", "chicken", "turkey", "fish", "shrimp", "bacon", "sausage", "ham"]
        dairy_terms = ["milk", "cheese", "butter", "yogurt", "cream", "ghee"]
        gluten_terms = ["wheat", "flour", "bread", "pasta", "noodle", "barley", "rye", "cracker"]
        nuts = ["almond", "peanut", "walnut", "pecan", "cashew", "hazelnut", "pistachio"]

        has_meat = any(m in ing_text for m in meats)
        has_dairy = any(d in ing_text for d in dairy_terms)
        has_gluten = any(g in ing_text for g in gluten_terms)
        has_nuts = any(n in ing_text for n in nuts)

        default_flags["vegan"] = not has_meat and not has_dairy and "egg" not in ing_text and "honey" not in ing_text
        default_flags["vegetarian"] = not has_meat
        default_flags["dairy_free"] = not has_dairy
        default_flags["gluten_free"] = not has_gluten
        default_flags["nut_free"] = not has_nuts
        # crude keto-friendly: low-carb keywords
        default_flags["keto_friendly"] = (
            default_flags["vegan"] is False and not any(w in ing_text for w in ["sugar", "flour", "rice", "potato"]) and has_meat
        )
        return default_flags


def audit_database_quality(conn, cursor):
    """Audit all recipes in the database and replace poor images"""
    print(f"\n=== STARTING DATABASE QUALITY AUDIT ===")
    
    # Get all recipes
    cursor.execute("SELECT id, title, image_url FROM recipes ORDER BY id")
    recipes = cursor.fetchall()
    
    replacements_made = 0
    
    for recipe in recipes:
        recipe_id, title, current_image_url = recipe['id'], recipe['title'], recipe['image_url']
        
        print(f"\nAuditing Recipe {recipe_id}: {title}")
        
        # Get current used URLs to avoid duplicates, but exclude this recipe's current image
        cursor.execute("SELECT image_url FROM recipes WHERE id != ?", (recipe_id,))
        used_urls = {row['image_url'] for row in cursor.fetchall()}
        
        # Audit current image
        current_score = audit_recipe_image_with_gemini(title, current_image_url)
        print(f"  Current image score: {current_score}/10")
        
        if current_score < 5:
            print(f"  Score too low ({current_score}/10). Searching for better image...")
            
            # Try to find a better image (excluding all other recipes' images)
            new_image_url, new_score, _ = find_best_image(title, used_urls)
            
            if new_image_url and new_score > current_score:
                # Update the database with the better image
                cursor.execute("UPDATE recipes SET image_url = ?, image_score = ? WHERE id = ?", 
                             (new_image_url, new_score, recipe_id))
                conn.commit()
                replacements_made += 1
                print(f"  ✓ Replaced with better image (Score: {new_score}/10)")
                
                # Update our used_urls set to include the new image and remove the old one
                used_urls.discard(current_image_url)  # Remove old image
                used_urls.add(new_image_url)          # Add new image
            else:
                print(f"  ✗ Could not find better image")
        else:
            print(f"  ✓ Current image is good enough (Score: {current_score}/10)")
        
        time.sleep(1)  # Be gentle on APIs during audit
    
    print(f"\n=== AUDIT COMPLETE ===")
    print(f"Replaced {replacements_made} poor images out of {len(recipes)} recipes.")
    return replacements_made

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
    print(f"Quality audits will run every {AUDIT_INTERVAL_HOURS} hours.")
    
    last_audit_time = time.time()
    
    while current_recipe_count < TARGET_RECIPE_COUNT:
        current_time = time.time()
        
        # Check if it's time for a quality audit
        if current_time - last_audit_time >= (AUDIT_INTERVAL_HOURS * 3600):
            audit_database_quality(conn, cursor)
            last_audit_time = current_time
            
            # Refresh used URLs after audit
            cursor.execute("SELECT image_url FROM recipes")
            used_image_urls = {row['image_url'] for row in cursor.fetchall()}
        
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

                # Nutrition and flags
                ingredients_list = recipe.get('ingredients', [])
                nutrition = estimate_nutrition(ingredients_list, title)
                flags = classify_dietary_flags(ingredients_list, title)
                
                if image_url and score >= MINIMUM_IMAGE_SCORE:
                    print(f"  --> SUCCESS! Recipe '{title}' passed quality gate with score {score}.")
                    recipe_data = {
                        "title": title,
                        "description": recipe.get('description', ''),
                        "cuisine": recipe.get('cuisine', 'Unknown'),
                        "difficulty": recipe.get('difficulty', 'Unknown'),
                        "dish_type": dish_type,
                        "ingredients_json": json.dumps(ingredients_list),
                        "instructions_json": json.dumps(recipe.get('instructions', [])),
                        "image_url": image_url,
                        "image_score": score,
                        "theme": theme,
                        "nutrition_json": json.dumps(nutrition),
                        "vegan": 1 if flags.get('vegan') else 0,
                        "vegetarian": 1 if flags.get('vegetarian') else 0,
                        "gluten_free": 1 if flags.get('gluten_free') else 0,
                        "dairy_free": 1 if flags.get('dairy_free') else 0,
                        "nut_free": 1 if flags.get('nut_free') else 0,
                        "keto_friendly": 1 if flags.get('keto_friendly') else 0,
                    }
                    insert_recipe(cursor, recipe_data)
                    used_image_urls.add(image_url)
                    conn.commit()
                    current_recipe_count += 1
                else:
                    print(f"  --> SKIPPED. Recipe '{title}' did not meet quality gate (Score: {score}) or no unique image was found.")
                
                time.sleep(2)
            
    conn.close()
    print(f"\n--- Generation Complete! ---")
    print(f"Successfully collected and stored {current_recipe_count} high-quality recipes in recipes.db.")

if __name__ == "__main__":
    main()
