import os
import json
import time
import sqlite3
import requests
from dotenv import load_dotenv

# Local imports from recipeGenerator for shared logic
from recipeGenerator import (
    load_dotenv as _rg_load,  # ensure .env is loaded in main
    estimate_nutrition,
    classify_dietary_flags,
    find_best_image,
    setup_db,
)


def open_db(path: str = 'recipes.db'):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return conn, cur


def update_recipe(cur: sqlite3.Cursor, recipe_id: int, updates: dict):
    keys = [
        'image_url', 'image_score', 'nutrition_json', 'vegan', 'vegetarian',
        'gluten_free', 'dairy_free', 'nut_free', 'keto_friendly'
    ]
    set_parts = []
    values = []
    for k in keys:
        if k in updates:
            set_parts.append(f"{k} = ?")
            values.append(updates[k])
    if not set_parts:
        return
    values.append(recipe_id)
    sql = f"UPDATE recipes SET {', '.join(set_parts)} WHERE id = ?"
    cur.execute(sql, values)


def fix_all_recipes():
    load_dotenv()
    # Ensure DB has latest columns
    conn_setup, cur_setup = setup_db()
    conn_setup.close()

    conn, cur = open_db()
    cur.execute("SELECT COUNT(*) FROM recipes")
    total = cur.fetchone()[0]
    print(f"=== Starting fix_recipes on {total} recipes ===")

    cur.execute("SELECT id, title, ingredients_json, image_url FROM recipes ORDER BY id")
    rows = cur.fetchall()

    used_urls = set()
    cur.execute("SELECT image_url FROM recipes")
    used_urls = {row[0] for row in cur.fetchall() if row[0]}

    # Track counts to detect duplicates
    url_counts = {}
    for url in used_urls:
        url_counts[url] = url_counts.get(url, 0) + 1

    processed = 0
    for row in rows:
        rid = row['id']
        title = row['title'] or ''
        ingredients = []
        try:
            ingredients = json.loads(row['ingredients_json'] or '[]')
        except Exception:
            ingredients = []

        print(f"\n--- [{processed+1}/{total}] Auditing Recipe {rid}: {title} ---")

        current_url = row['image_url']
        is_duplicate = current_url and url_counts.get(current_url, 0) > 1

        # Always search for a better or unique image; allow below threshold if resolving duplicates/no image
        allow_below = is_duplicate or not current_url
        best_url, best_score, _dish = find_best_image(
            title,
            used_urls - ({current_url} if current_url else set()),
            allow_below_threshold=allow_below,
        )

        nutrition = estimate_nutrition(ingredients, title)
        flags = classify_dietary_flags(ingredients, title)

        updates = {
            'nutrition_json': json.dumps(nutrition),
            'vegan': 1 if flags.get('vegan') else 0,
            'vegetarian': 1 if flags.get('vegetarian') else 0,
            'gluten_free': 1 if flags.get('gluten_free') else 0,
            'dairy_free': 1 if flags.get('dairy_free') else 0,
            'nut_free': 1 if flags.get('nut_free') else 0,
            'keto_friendly': 1 if flags.get('keto_friendly') else 0,
        }

        replaced_image = False
        if best_url and (not current_url or is_duplicate or best_score):
            # Replace if duplicate, missing, or better image found
            updates['image_url'] = best_url
            updates['image_score'] = best_score
            replaced_image = True

        if replaced_image:
            # Maintain global uniqueness accounting
            if current_url and url_counts.get(current_url, 0) > 0:
                url_counts[current_url] -= 1
                if url_counts[current_url] <= 0:
                    url_counts.pop(current_url, None)
            used_urls.add(best_url)
            url_counts[best_url] = url_counts.get(best_url, 0) + 1
            print(f"  ✓ Updated image (score {best_score})")
        else:
            # Ensure current is counted
            if current_url:
                url_counts[current_url] = url_counts.get(current_url, 0) + 0  # no-op ensure key exists
            print("  ↷ Kept existing image")

        update_recipe(cur, rid, updates)
        conn.commit()

        processed += 1
        time.sleep(1)

    conn.close()
    print("\n=== fix_recipes complete ===")


if __name__ == '__main__':
    fix_all_recipes()


