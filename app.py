from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import uuid
import json
import re
import os
import sys
import time
from dotenv import load_dotenv

# ==========================
# LOGGER — writes to console AND server.log simultaneously
# ==========================
class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(os.path.join(os.path.dirname(__file__), "server.log"), "w", encoding="utf-8")
sys.stdout = TeeLogger(sys.__stdout__, _log_file)
sys.stderr = TeeLogger(sys.__stderr__, _log_file)

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==========================
# GROQ CONFIG
# ==========================

GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    raise RuntimeError(
        "No Groq API key found. Set GROQ_API_KEY in your .env file."
    )

print(f"[STARTUP] Loaded Groq API key: {GROQ_KEY[:8]}...{GROQ_KEY[-4:]}")

GROQ_CLIENT = Groq(api_key=GROQ_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"

print(f"[STARTUP] Groq client ready, model={MODEL_NAME}")

def get_next_client():
    """Returns the Groq client (single key, stateless)."""
    print(f"[LLM] Using Groq client, model={MODEL_NAME}")
    return GROQ_CLIENT

# ==========================
# DB CONFIG
# ==========================

DB_FILE = "db.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {"users": {}}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ==========================
# UTIL FUNCTIONS
# ==========================

def clean_ingredients(ingredients):
    """Clean ingredient list — split on commas only, preserve multi-word items like 'chicken breast'."""
    if isinstance(ingredients, str):
        ingredients = [i.strip() for i in ingredients.split(",") if i.strip()]

    # Expand each item by splitting on commas only (not spaces)
    expanded = []
    for i in ingredients:
        parts = [p.strip() for p in str(i).split(",")]
        expanded.extend(parts)

    cleaned = []
    for item in expanded:
        item = item.strip().strip(",").strip().lower()
        if item and item != ",":
            cleaned.append(item)
    return cleaned


# ==========================
# PROMPT
# ==========================

CUISINE_STYLES = {
    "Indian": ["curry", "fry", "masala", "biryani", "tikka"],
    "Chinese": ["stir fry", "noodle dish", "fried rice", "dumpling", "soup"],
    "Italian": ["pasta", "risotto", "baked dish", "bruschetta", "salad"],
    "Mexican": ["tacos", "burrito bowl", "enchilada", "quesadilla", "salsa bowl"],
    "Thai": ["curry", "stir fry", "noodle soup", "salad", "fried rice"],
    "French": ["sauté", "gratin", "ratatouille", "crêpe", "casserole"],
    "Japanese": ["ramen", "donburi", "teriyaki", "stir fry", "tempura"],
    "Greek": ["souvlaki", "baked dish", "salad", "stew", "wrap"],
    "Spanish": ["paella", "tapas", "stew", "tortilla", "baked dish"],
    "British": ["pie", "roast", "stew", "bake", "casserole"],
    "Vietnamese": ["pho", "banh mi", "stir fry", "spring rolls", "noodle bowl"],
    "Korean": ["bibimbap", "stir fry", "stew", "fried rice", "pancake"],
    "Moroccan": ["tagine", "couscous dish", "stew", "salad", "soup"],
    "Filipino": ["adobo", "sinigang", "stir fry", "stew", "fried rice"],
}
DEFAULT_STYLES = ["stir fry", "curry", "baked dish", "soup", "salad"]

def build_prompt(ingredients, cuisine="Indian", diet="Balanced", allergies=None, variation=0):

    allergies = allergies or []
    styles = CUISINE_STYLES.get(cuisine, DEFAULT_STYLES)
    style_hint = styles[variation % len(styles)]

    return f"""You are a world-class professional chef and nutritionist with deep expertise in {cuisine} cuisine.

Ingredients available: {", ".join(ingredients)}
Cuisine: {cuisine}
Diet: {diet}
Allergies to avoid: {", ".join(allergies) if allergies else "None"}
Suggested style (use ONLY if it makes sense with the ingredients): {style_hint}

TASK: Create one authentic, detailed {cuisine} recipe using the ingredients above.

CRITICAL — DISH NAME SELECTION:
- First, think about what dish people MOST COMMONLY make with these exact ingredients in {cuisine} cuisine.
- For example: chicken + noodles → "Chicken Noodles" or "Chicken Hakka Noodles" (Indian), "Chicken Chow Mein" (Chinese). egg + rice → "Egg Fried Rice". paneer + spinach → "Palak Paneer". chicken + rice → "Chicken Biryani" (Indian), "Chicken Fried Rice" (Chinese).
- Choose the MOST NATURAL and POPULAR dish that uses these ingredients together. Do NOT invent hybrid names like "Chicken Curry Noodle Biryani" — that is not a real dish.
- The suggested style "{style_hint}" is only a hint. If the ingredients naturally form a different dish (e.g., chicken + noodles = noodle dish, NOT a curry), IGNORE the style hint and use the correct dish type.

MANDATORY RULES:
1. The dish MUST be an authentic {cuisine} recipe that people actually cook and eat. Use real dish names from {cuisine} cuisine.
2. Use cooking techniques, spices, sauces and seasonings that are traditional to {cuisine} cuisine.
3. Include all necessary ingredients with exact quantities, using pantry staples typical of {cuisine} cooking.
4. Every process step MUST be specific with actions, timings and quantities. Include at least 10 detailed steps.
5. Respect allergies strictly. Match the diet type: {diet}.
6. Nutrition MUST list per-serving values for: Calories, Protein, Carbohydrates, Fat, Fiber, Sodium.
7. Benefits MUST be 4-5 detailed sentences explaining specific health benefits of the main ingredients.
8. Include at least 4 practical chef tips specific to this dish.
9. Pairing suggestions should match {cuisine} cuisine traditions.

OUTPUT: Return ONLY a raw JSON object. No markdown, no backticks, no explanation before or after the JSON.
All array items MUST be plain strings — NOT objects or numbered keys.

Exact JSON structure:

{{
  "name": "<real {cuisine} dish name that people actually search for>",
  "cuisine": "{cuisine}",
  "dietType": "{diet}",
  "ingredients": [
    "<quantity> <ingredient 1>",
    "<quantity> <ingredient 2>"
  ],
  "prepTime": "<X minutes>",
  "cookTime": "<X minutes>",
  "servings": "<N people>",
  "difficulty": "<Easy|Medium|Hard>",
  "process": [
    "<detailed step 1 with timing and quantities>",
    "<detailed step 2>"
  ],
  "nutrition": "Calories: ~XXX kcal\\nProtein: XXg\\nCarbohydrates: XXg\\nFat: XXg\\nFiber: Xg\\nSodium: XXXmg",
  "benefits": "<4-5 sentences about health benefits of key ingredients>",
  "tips": [
    "<practical tip 1>",
    "<practical tip 2>"
  ],
  "pairing": "<2-3 sentences about what to serve with this dish, matching {cuisine} traditions>"
}}"""


# ==========================
# NORMALIZER HELPERS
# ==========================

def normalize_list_of_strings(data):
    """Convert a list of strings or objects into a clean list of strings."""
    if not data or not isinstance(data, list):
        return []
    result = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
        elif isinstance(item, dict):
            # Try common key names Gemini might use for step/ingredient objects
            for key in ["description", "instruction", "step", "text", "content", "detail", "name", "item"]:
                if key in item and isinstance(item[key], str):
                    result.append(item[key].strip())
                    break
            else:
                # Fallback: join all non-empty string values
                parts = [str(v) for v in item.values() if v and isinstance(v, str)]
                if parts:
                    result.append(" — ".join(parts))
        else:
            text = str(item).strip()
            if text:
                result.append(text)
    return result


def normalize_nutrition(nutrition):
    """Normalize nutrition field — could be a plain string or a dict."""
    if not nutrition:
        return ""
    if isinstance(nutrition, str):
        return nutrition.strip()
    if isinstance(nutrition, dict):
        key_map = {
            "calories": "Calories", "protein": "Protein",
            "carbohydrates": "Carbohydrates", "carbs": "Carbohydrates",
            "fat": "Fat", "fiber": "Fiber", "sodium": "Sodium",
            "sugar": "Sugar", "cholesterol": "Cholesterol"
        }
        lines = []
        for k, v in nutrition.items():
            label = key_map.get(k.lower(), k.replace("_", " ").title())
            lines.append(f"{label}: {v}")
        return "\n".join(lines)
    return str(nutrition)


def extract_calories(nutrition_text):
    """Extract numeric calories value from a nutrition string like 'Calories: ~420 kcal'."""
    if not nutrition_text:
        return None
    match = re.search(r"[Cc]alories[:\s]*~?(\d+)", nutrition_text)
    return int(match.group(1)) if match else None


# ==========================
# AI RECIPE GENERATOR
# ==========================

def generate_recipe(ingredients, cuisine="Indian", diet="Balanced", allergies=None, variation=0):

    prompt = build_prompt(ingredients, cuisine, diet, allergies, variation)
    print(f"\n{'='*60}")
    print(f"[GENERATE] Recipe #{variation+1} | ingredients={ingredients} | cuisine={cuisine} | diet={diet} | allergies={allergies}")
    print(f"[GENERATE] Prompt length: {len(prompt)} chars")
    print(f"[GENERATE] Model: {MODEL_NAME}")

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                wait = attempt * 2
                print(f"[LLM] Retry attempt {attempt+1}/{max_attempts} with next API key, waiting {wait}s...")
                time.sleep(wait)

            print(f"[LLM] Attempt {attempt+1}/{max_attempts} — calling Groq...")
            start_time = time.time()
            client = get_next_client()
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            elapsed = round(time.time() - start_time, 2)
            print(f"[LLM] Groq responded in {elapsed}s")

            text = completion.choices[0].message.content
            # Check for blocked/empty responses
            if not text:
                raise Exception("Empty response from Groq API")

            text = text.strip()
            print(f"[LLM] Response length: {len(text)} chars")
            print(f"[LLM] Response preview: {text[:200]}...")

            # Strip markdown code fences Gemini sometimes wraps responses in
            if "```" in text:
                fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
                if fence_match:
                    text = fence_match.group(1).strip()
                    print(f"[LLM] Stripped code fences, cleaned length: {len(text)}")

            # Extract JSON object from the response
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                print(f"[LLM] ERROR: No JSON object found in response")
                print(f"[LLM] Full response text:\n{text[:500]}")
                raise Exception("No JSON object found in AI response")

            json_text = match.group(0)
            print(f"[LLM] JSON extracted, length: {len(json_text)}")
            data = json.loads(json_text)
            print(f"[LLM] JSON parsed successfully, keys: {list(data.keys())}")

            # Normalize all fields — Gemini sometimes returns objects/dicts instead of strings
            process_steps   = normalize_list_of_strings(data.get("process", []))
            ingredient_list = normalize_list_of_strings(data.get("ingredients", []))
            tips_list       = normalize_list_of_strings(data.get("tips", []))
            nutrition_text  = normalize_nutrition(data.get("nutrition", ""))

            benefits_raw = data.get("benefits", "")
            if isinstance(benefits_raw, list):
                benefits_text = " ".join(normalize_list_of_strings(benefits_raw))
            elif isinstance(benefits_raw, dict):
                benefits_text = " ".join(str(v) for v in benefits_raw.values() if isinstance(v, str))
            else:
                benefits_text = str(benefits_raw).strip()

            recipe_name = str(data.get("name", "")).strip()
            print(f"[LLM] SUCCESS — Recipe name: '{recipe_name}'")
            print(f"[LLM]   ingredients: {len(ingredient_list)}, process steps: {len(process_steps)}, tips: {len(tips_list)}")
            print(f"[LLM]   nutrition: {nutrition_text[:80]}..." if len(nutrition_text) > 80 else f"[LLM]   nutrition: {nutrition_text}")
            print(f"{'='*60}\n")

            return {
                "id": str(uuid.uuid4()),
                "name": recipe_name,
                "cuisine": str(data.get("cuisine", cuisine)).strip(),
                "dietType": str(data.get("dietType", diet)).strip(),
                "ingredients": ingredient_list if ingredient_list else list(ingredients),
                "prepTime": str(data.get("prepTime", "15 minutes")),
                "cookTime": str(data.get("cookTime", "25 minutes")),
                "servings": str(data.get("servings", "2-3 people")),
                "difficulty": str(data.get("difficulty", "Medium")),
                "process": process_steps,
                "nutrition": nutrition_text,
                "benefits": benefits_text,
                "tips": tips_list,
                "pairing": str(data.get("pairing", "")).strip(),
                "calories": extract_calories(nutrition_text)
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[LLM] FAILED attempt {attempt+1}/{max_attempts}: {type(e).__name__}: {error_msg}")
            if attempt < max_attempts - 1:
                print(f"[LLM] Will retry with next API key...")
                continue
            import traceback
            traceback.print_exc()
            print(f"[FALLBACK] All {max_attempts} LLM attempts failed — switching to fallback recipe")
            print(f"{'='*60}\n")
            return generate_fallback_recipe(ingredients, cuisine=cuisine, diet=diet, variation=variation)


# ==========================
# FALLBACK RECIPE
# ==========================

CUISINE_FALLBACK = {
    "Indian": {
        "styles": ["Curry", "Fry", "Masala"],
        "staples": ["2 tbsp cooking oil", "1 medium onion, finely chopped", "1 tbsp ginger-garlic paste",
                     "1 tsp cumin seeds", "1 tsp turmeric powder", "1 tsp red chilli powder",
                     "1 tsp coriander powder", "1/2 tsp garam masala", "salt to taste", "fresh coriander for garnish"],
        "pairing": "steamed basmati rice or freshly made chapati with cucumber raita on the side",
    },
    "Chinese": {
        "styles": ["Stir Fry", "Fried Rice", "Noodles"],
        "staples": ["2 tbsp sesame oil", "3 cloves garlic, minced", "1 inch ginger, julienned",
                     "2 tbsp soy sauce", "1 tbsp oyster sauce", "1 tsp cornstarch",
                     "1 spring onion, sliced", "pinch white pepper", "salt to taste"],
        "pairing": "steamed jasmine rice or hot wonton soup on the side",
    },
    "Italian": {
        "styles": ["Pasta", "Baked Dish", "Risotto"],
        "staples": ["2 tbsp extra virgin olive oil", "3 cloves garlic, minced", "1 medium onion, diced",
                     "1 can crushed tomatoes", "1 tsp dried oregano", "1 tsp dried basil",
                     "1/4 cup parmesan cheese", "salt and black pepper to taste", "fresh basil for garnish"],
        "pairing": "garlic bread and a fresh green salad with balsamic vinaigrette",
    },
    "Mexican": {
        "styles": ["Tacos", "Burrito Bowl", "Quesadilla"],
        "staples": ["2 tbsp vegetable oil", "1 medium onion, diced", "3 cloves garlic, minced",
                     "1 tsp cumin powder", "1 tsp chili powder", "1/2 tsp smoked paprika",
                     "juice of 1 lime", "salt to taste", "fresh cilantro for garnish"],
        "pairing": "tortilla chips with guacamole and a refreshing agua fresca",
    },
    "Thai": {
        "styles": ["Curry", "Stir Fry", "Noodle Soup"],
        "staples": ["2 tbsp vegetable oil", "2 tbsp Thai curry paste", "1 can coconut milk",
                     "2 tbsp fish sauce", "1 tbsp palm sugar", "Thai basil leaves",
                     "2 kaffir lime leaves", "1 stalk lemongrass", "salt to taste"],
        "pairing": "steamed jasmine rice and a side of fresh Thai papaya salad",
    },
    "Japanese": {
        "styles": ["Teriyaki", "Donburi", "Stir Fry"],
        "staples": ["2 tbsp vegetable oil", "3 tbsp soy sauce", "2 tbsp mirin",
                     "1 tbsp sake", "1 tsp sugar", "1 tsp sesame seeds",
                     "1 spring onion, sliced", "1 tsp grated ginger", "salt to taste"],
        "pairing": "steamed short-grain rice, miso soup, and pickled ginger on the side",
    },
}
DEFAULT_FALLBACK = {
    "styles": ["Stir Fry", "Baked Dish", "Stew"],
    "staples": ["2 tbsp cooking oil", "1 medium onion, diced", "3 cloves garlic, minced",
                 "salt and pepper to taste", "fresh herbs for garnish"],
    "pairing": "a fresh side salad and crusty bread",
}

def generate_fallback_recipe(ingredients, cuisine="Indian", diet="Balanced", variation=0):

    main = ingredients[0] if ingredients else "vegetables"
    fb = CUISINE_FALLBACK.get(cuisine, DEFAULT_FALLBACK)
    styles = fb["styles"]
    style = styles[variation % len(styles)]
    staples = fb["staples"]
    pairing_hint = fb["pairing"]

    full_ingredients = list(ingredients) + staples

    # Detect diet type
    all_ingredients_lower = " ".join(ingredients).lower()
    meat_keywords = ["chicken", "mutton", "fish", "prawn", "shrimp", "beef", "pork", "lamb"]
    egg_keywords = ["egg", "eggs"]
    has_meat = any(k in all_ingredients_lower for k in meat_keywords)
    has_egg = any(k in all_ingredients_lower for k in egg_keywords)

    if diet and diet not in ("", "Balanced"):
        diet_type = diet
    elif has_meat or has_egg:
        diet_type = "Non Vegetarian"
    else:
        diet_type = "Vegetarian"

    # Build dish name
    key_ingredients = [i.title() for i in ingredients[:2]]
    dish_name = f"{' '.join(key_ingredients)} {style}"

    process = [
        f"Prepare the {main} by washing and cutting into appropriate sized pieces.",
        f"Prepare all other ingredients — dice onions, mince garlic, measure out sauces and spices.",
        f"Heat oil in a suitable pan over medium-high heat.",
        f"Add aromatics (onion, garlic) and cook for 2-3 minutes until fragrant.",
        f"Add the {main} and cook for 5-7 minutes, stirring occasionally.",
        f"Add seasonings and sauces. Mix well to coat evenly.",
        f"Cook for another 8-10 minutes until {main} is fully cooked through.",
        f"Taste and adjust seasoning with salt and pepper.",
        f"Garnish with fresh herbs and serve immediately.",
        f"Enjoy with {pairing_hint}."
    ]

    return {
        "id": str(uuid.uuid4()),
        "name": dish_name,
        "cuisine": cuisine,
        "dietType": diet_type,
        "ingredients": full_ingredients,
        "prepTime": "15 minutes",
        "cookTime": "25 minutes",
        "servings": "2-3 people",
        "difficulty": "Easy",
        "process": process,
        "nutrition": "Calories: ~300 kcal\nProtein: 12g\nCarbohydrates: 28g\nFat: 14g\nFiber: 4g\nSodium: 500mg",
        "benefits": (
            f"The main ingredients provide essential nutrients including protein, vitamins and minerals. "
            f"Garlic and onions are rich in antioxidants that support immune function and cardiovascular health. "
            f"Cooking with quality oil provides healthy fats important for nutrient absorption. "
            f"Fresh herbs add flavor without extra sodium and contain beneficial plant compounds. "
            f"A balanced combination of ingredients supports energy metabolism and overall wellness."
        ),
        "tips": [
            "Always heat the pan properly before adding ingredients for the best sear and flavor.",
            "Taste and adjust seasoning at the end — flavors concentrate during cooking.",
            f"Cut the {main} into uniform pieces for even cooking.",
            "Let the dish rest for a few minutes before serving to allow flavors to meld.",
            "Use fresh ingredients whenever possible for the best flavor and nutrition."
        ],
        "pairing": f"{dish_name} pairs well with {pairing_hint}.",
        "calories": 300
    }


# ==========================
# API - RECIPE GENERATION
# ==========================

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    print(f"\n{'#'*60}")
    print(f"[API] POST /recommend — raw payload: {json.dumps(data, default=str)[:500]}")

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    ingredients = clean_ingredients(data.get("ingredients", []))
    cuisine = data.get("cuisine", "Indian")
    diet = data.get("diet_goal", "Balanced")
    allergies = data.get("allergies", [])
    top_n = data.get("top_n", 1)

    print(f"[API] Cleaned ingredients: {ingredients}")
    print(f"[API] cuisine={cuisine}, diet={diet}, allergies={allergies}, top_n={top_n}")

    if not ingredients:
        print(f"[API] ERROR: No ingredients after cleaning")
        return jsonify({"error": "ingredients required"}), 400

    recipes = []
    for i in range(top_n):
        print(f"\n[API] Generating recipe {i+1}/{top_n}...")
        recipe = generate_recipe(ingredients=ingredients, cuisine=cuisine, diet=diet, allergies=allergies, variation=i)
        source = "LLM" if recipe.get("difficulty") != "Easy" else "FALLBACK"
        print(f"[API] Recipe {i+1} done: '{recipe.get('name')}' (source: likely {source})")
        recipes.append(recipe)

    print(f"[API] Returning {len(recipes)} recipe(s)")
    print(f"{'#'*60}\n")
    return jsonify(recipes)


# ==========================
# API - PROFILE RECOMMENDATIONS
# ==========================

@app.route("/api/recommendations/<email>", methods=["GET"])
def get_personalized_recommendations(email):
    db = load_db()
    user = db["users"].get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    profile = user["profile"]
    fav_ingredients = profile.get("favoriteIngredients", [])
    cuisines = profile.get("cuisinePreferences", [])
    allergies = profile.get("allergies", [])
    diet_goal = profile.get("dietType", None)

    # Fetch all past generated recipes from search history
    search_history = user.get("searchHistory", [])
    past_ingredients = []
    for batch in search_history:
        if isinstance(batch, list):
            past_ingredients.extend(batch)

    # Build ingredients query: prefer past search ingredients, fall back to profile
    if past_ingredients:
        ingredients = clean_ingredients(list(dict.fromkeys(past_ingredients)))  # deduplicated
    elif fav_ingredients:
        ingredients = clean_ingredients(fav_ingredients)
    else:
        ingredients = ["vegetables"]

    target_cuisine = cuisines[0] if cuisines else "Indian"

    # Generate 3 personalised recipes
    new_recipes = []
    for i in range(3):
        recipe = generate_recipe(
            ingredients=ingredients,
            cuisine=target_cuisine,
            diet=diet_goal or "Balanced",
            allergies=allergies,
            variation=i
        )
        new_recipes.append(recipe)

    return jsonify(new_recipes)


# ==========================
# API - AUTH
# ==========================

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    email = data.get("email")
    name = data.get("name")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    db = load_db()
    if email in db["users"]:
        return jsonify({"error": "User already exists"}), 400

    db["users"][email] = {
        "name": name,
        "password": password,
        "profile": {
            "name": name,
            "email": email,
            "dietGoals": [],
            "dietType": "",
            "cuisinePreferences": [],
            "allergies": [],
            "favoriteIngredients": [],
            "favoriteVegetables": [],
            "favoriteDishes": []
        },
        "favorites": [],
        "searchHistory": []
    }
    save_db(db)
    return jsonify({"message": "User created", "user": db["users"][email]["profile"]})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    db = load_db()
    user = db["users"].get(email)

    if not user or user["password"] != password:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "user": user["profile"],
        "favorites": user["favorites"],
        "searchHistory": user.get("searchHistory", [])
    })


# ==========================
# API - PROFILE
# ==========================

@app.route("/api/profile/<email>", methods=["GET", "PUT"])
def profile(email):
    db = load_db()
    user = db["users"].get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if request.method == "GET":
        return jsonify({
            "profile": user["profile"],
            "searchHistory": user.get("searchHistory", [])
        })

    if request.method == "PUT":
        update_data = request.json
        user["profile"].update(update_data)
        save_db(db)
        return jsonify(user["profile"])


# ==========================
# API - SEARCH HISTORY
# ==========================

@app.route("/api/search-history/<email>", methods=["POST"])
def save_search_history(email):
    db = load_db()
    user = db["users"].get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    history_batch = request.json
    if not isinstance(history_batch, list):
        return jsonify({"error": "Invalid data format"}), 400

    current_history = user.get("searchHistory", [])
    new_history = [history_batch] + current_history
    user["searchHistory"] = new_history[:5]

    save_db(db)
    return jsonify(user["searchHistory"])


# ==========================
# API - FAVORITES
# ==========================

@app.route("/api/favorites/<email>", methods=["GET", "POST"])
def favorites(email):
    db = load_db()
    user = db["users"].get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if request.method == "GET":
        return jsonify(user["favorites"])

    if request.method == "POST":
        recipe = request.json
        if recipe not in user["favorites"]:
            user["favorites"].append(recipe)
            save_db(db)
        return jsonify(user["favorites"])


@app.route("/api/favorites/<email>/<recipe_id>", methods=["DELETE"])
def delete_favorite(email, recipe_id):
    db = load_db()
    user = db["users"].get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    user["favorites"] = [f for f in user["favorites"] if str(f.get("recipe", {}).get("id")) != str(recipe_id)]
    save_db(db)
    return jsonify(user["favorites"])


# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
