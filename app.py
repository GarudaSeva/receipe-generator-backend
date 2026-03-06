from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import uuid
import json
import re
import os
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==========================
# GEMINI CONFIG - ROUND ROBIN KEY ROTATION
# ==========================

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]
# Remove any None keys
API_KEYS = [k for k in API_KEYS if k]

_current_key_index = 0

def get_next_model():
    """Round-robin: returns a fresh model using the next API key each time."""
    global _current_key_index
    key = API_KEYS[_current_key_index % len(API_KEYS)]
    _current_key_index += 1
    genai.configure(api_key=key)
    print(f"[DEBUG] Using API key #{(_current_key_index - 1) % len(API_KEYS) + 1}")
    return genai.GenerativeModel("gemini-2.0-flash")

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
    if isinstance(ingredients, str):
        ingredients = re.split(r"[,\s]+", ingredients)

    # Strip commas AND whitespace from each item, then filter empty ones
    cleaned = []
    for i in ingredients:
        item = str(i).strip().strip(",").strip().lower()
        if item and item != ",":
            cleaned.append(item)
    return cleaned


# ==========================
# PROMPT
# ==========================

DISH_STYLES = ["curry", "fry", "gravy", "masala", "biryani"]

def build_prompt(ingredients, cuisine="Indian", diet="Balanced", allergies=None, variation=0):

    allergies = allergies or []

    # Pick a different style hint for each variation
    style_hint = DISH_STYLES[variation % len(DISH_STYLES)]

    return f"""
You are an expert Indian home chef.

Ingredients available:
{", ".join(ingredients)}

Cuisine:
{cuisine}

Diet:
{diet}

Avoid allergies:
{", ".join(allergies) if allergies else "None"}

Your task:

1. Identify the MOST logical Indian dish using these ingredients.
2. Prefer authentic home style cooking.
3. You MUST also use common Indian kitchen staples in the recipe: cooking oil, onions, garlic, ginger, green chillies, curry leaves, mustard seeds, cumin seeds, turmeric powder, red chilli powder, coriander powder, garam masala, salt, and fresh coriander for garnish. Include these in the ingredients list.
4. Make a {style_hint} style dish. Do NOT repeat the same dish type if called multiple times.

Rules:
- Choose a REAL dish name like "Chicken Curry", "Aloo Gobi", "Mutton Keema" etc.
- Cooking steps must be practical and clear.
- Do NOT use vague sentences.
- The ingredients list MUST include all spices and oil used, not just the main items.

GOOD step example:
"Heat 2 tbsp oil in a kadai. Add mustard seeds and allow them to splutter."

BAD step example:
"Prepare ingredients and cook."

Return JSON only.

Format:

{{
"name": "",
"cuisine": "",
"dietType": "",
"ingredients": [],
"prepTime": "",
"cookTime": "",
"servings": "",
"difficulty": "",
"process": [],
"nutrition": "",
"benefits": "",
"tips": [],
"pairing": ""
}}
"""


# ==========================
# AI RECIPE GENERATOR
# ==========================

def generate_recipe(ingredients, cuisine="Indian", diet="Balanced", allergies=None, variation=0):

    prompt = build_prompt(ingredients, cuisine, diet, allergies, variation)
    print(f"[DEBUG] Generating recipe #{variation+1} for: {ingredients}")

    # Retry up to 2 times with delay for rate limits
    for attempt in range(2):
        try:
            if attempt > 0:
                print(f"[DEBUG] Retry with next API key, waiting 2 seconds...")
                time.sleep(2)

            current_model = get_next_model()
            response = current_model.generate_content(prompt)
            text = response.text.strip()
            print(f"[DEBUG] AI Response received, length: {len(text)}")

            match = re.search(r"\{.*\}", text, re.DOTALL)

            if match:
                data = json.loads(match.group(0))
            else:
                print(f"[DEBUG] No JSON found in response: {text[:200]}")
                raise Exception("Invalid JSON")

            return {
                "id": str(uuid.uuid4()),
                "name": data.get("name"),
                "cuisine": data.get("cuisine", cuisine),
                "dietType": data.get("dietType", diet),
                "ingredients": data.get("ingredients", ingredients),
                "prepTime": data.get("prepTime", "15 min"),
                "cookTime": data.get("cookTime", "25 min"),
                "servings": data.get("servings", "2"),
                "difficulty": data.get("difficulty", "Easy"),
                "process": data.get("process", []),
                "nutrition": data.get("nutrition", ""),
                "benefits": data.get("benefits", ""),
                "tips": data.get("tips", []),
                "pairing": data.get("pairing", "")
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] AI Error (attempt {attempt+1}): {type(e).__name__}: {error_msg}")
            # If rate limited and first attempt, retry with next key
            if ("429" in error_msg or "ResourceExhausted" in str(type(e).__name__)) and attempt == 0:
                continue
            import traceback
            traceback.print_exc()
            return generate_fallback_recipe(ingredients, variation)


# ==========================
# FALLBACK RECIPE
# ==========================

FALLBACK_DISHES = [
    {
        "suffix": "Curry",
        "process": [
            "Wash and cut the {main} into medium sized pieces.",
            "Heat 2 tablespoons oil in a deep kadai on medium flame.",
            "Add 1 tsp mustard seeds, 1 tsp cumin seeds and let them splutter.",
            "Add curry leaves, 2 dried red chillies and a pinch of hing (asafoetida).",
            "Add 1 large finely chopped onion and sauté for 5-7 minutes until golden brown.",
            "Add 1 tbsp ginger-garlic paste and sauté for 2 minutes until raw smell goes away.",
            "Add the {main} pieces, 1 tsp turmeric, 1 tsp red chilli powder, 1 tsp coriander powder and salt.",
            "Mix well, add 1 cup water, cover and cook on low flame for 15-20 minutes.",
            "Add 1/2 tsp garam masala, stir well and cook for 2 more minutes.",
            "Turn off the stove. Garnish with fresh chopped coriander leaves. Serve hot with rice."
        ]
    },
    {
        "suffix": "Fry",
        "process": [
            "Wash and chop the {main} into thin slices or small pieces.",
            "Heat 3 tablespoons oil in a flat tawa or iron pan on medium-high flame.",
            "Add 1 tsp mustard seeds and wait for them to crackle.",
            "Add 8-10 curry leaves and 2 slit green chillies, sauté for 30 seconds.",
            "Add 1 medium sliced onion and fry for 4-5 minutes until edges turn brown.",
            "Add the {main} slices, spread evenly in the pan.",
            "Sprinkle 1/2 tsp turmeric, 1 tsp red chilli powder, and salt to taste.",
            "Cook on medium flame for 10-12 minutes, flipping occasionally for even crispiness.",
            "Increase flame to high for last 2 minutes to get crispy edges.",
            "Switch off stove. Squeeze half a lemon and garnish with chopped coriander. Serve hot."
        ]
    },
    {
        "suffix": "Masala",
        "process": [
            "Clean and cut the {main} into bite-sized cubes.",
            "Heat 2 tbsp oil + 1 tbsp ghee in a heavy bottom pan on medium flame.",
            "Add 1 tsp cumin seeds, 2 cloves, 1 small cinnamon stick and let them sizzle.",
            "Add 2 medium finely chopped onions and cook for 8-10 minutes until deep golden.",
            "Add 1 tbsp ginger-garlic paste, cook for 2 minutes until fragrant.",
            "Add 2 chopped tomatoes and cook until they become soft and oil separates (5-7 min).",
            "Add 1 tsp turmeric, 1.5 tsp red chilli powder, 1 tsp coriander powder, salt to taste.",
            "Add the {main} pieces and mix well with the masala. Cook for 3 minutes.",
            "Add 1/2 cup water, cover with lid and cook on low flame for 15 minutes until tender.",
            "Add 1 tsp garam masala, mix gently. Garnish with fresh coriander and serve with chapati."
        ]
    },
    {
        "suffix": "Gravy",
        "process": [
            "Wash and prepare the {main} by cutting into medium chunks.",
            "Soak 10 cashews in warm water for 10 minutes, then grind into smooth paste.",
            "Heat 2 tbsp oil in a deep pan on medium flame.",
            "Add 1 bay leaf, 2 green cardamom, 3 cloves and sauté for 30 seconds.",
            "Add 1 large finely chopped onion and cook for 6-8 minutes until soft and brown.",
            "Add 1 tbsp ginger-garlic paste and fry for 2 minutes.",
            "Add 2 pureed tomatoes, 1 tsp turmeric, 1 tsp chilli powder. Cook for 5 minutes.",
            "Add the {main} and cashew paste. Mix well. Add 1 cup water.",
            "Cover and simmer on low flame for 20 minutes until {main} is fully cooked.",
            "Finish with 1 tsp garam masala and cream (optional). Garnish with coriander. Serve with naan."
        ]
    },
    {
        "suffix": "Stir Fry",
        "process": [
            "Wash, peel and julienne-cut the {main} into thin strips.",
            "Heat 2 tbsp oil in a wok or large kadai on high flame.",
            "Add 1 tsp mustard seeds and 1/2 tsp cumin seeds, let them pop.",
            "Add 1 sprig curry leaves, 2 slit green chillies and 1 dried red chilli.",
            "Add 1 medium thinly sliced onion and toss on high heat for 2-3 minutes.",
            "Add the {main} strips immediately. Toss continuously on high flame.",
            "Add 1/4 tsp turmeric, 1/2 tsp red chilli powder and salt. Keep tossing.",
            "Add 1 tbsp freshly grated coconut and mix well.",
            "Cook for 5-7 minutes total on high heat until slightly charred but crunchy.",
            "Switch off flame. Squeeze lemon juice and garnish with coriander. Serve as a side dish."
        ]
    }
]

def generate_fallback_recipe(ingredients, variation=0):

    main = ingredients[0] if ingredients else "vegetables"
    dish = FALLBACK_DISHES[variation % len(FALLBACK_DISHES)]
    process = [step.format(main=main) for step in dish["process"]]

    # Build a proper ingredients list with kitchen staples
    full_ingredients = list(ingredients) + [
        "2 tbsp cooking oil",
        "1 medium onion",
        "1 tbsp ginger-garlic paste",
        "8-10 curry leaves",
        "2 green chillies",
        "1 tsp mustard seeds",
        "1 tsp cumin seeds",
        "1/2 tsp turmeric powder",
        "1 tsp red chilli powder",
        "salt to taste",
        "fresh coriander for garnish"
    ]

    return {
        "id": str(uuid.uuid4()),
        "name": f"{main.title()} {dish['suffix']}",
        "cuisine": "Indian",
        "dietType": "Vegetarian",
        "ingredients": full_ingredients,
        "prepTime": "10 min",
        "cookTime": "20 min",
        "servings": "2",
        "difficulty": "Easy",
        "process": process,
        "nutrition": "Rich in vitamins, minerals and fiber. Provides sustained energy and supports digestion.",
        "benefits": "Contains antioxidants from spices like turmeric and cumin. Supports immunity and healthy metabolism.",
        "tips": [
            "Cook on medium heat to avoid burning spices",
            "Use fresh vegetables for best flavor",
            "Adjust chilli powder to your spice tolerance"
        ],
        "pairing": "Serve with steamed rice or hot chapati"
    }


# ==========================
# API - RECIPE GENERATION
# ==========================

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    ingredients = clean_ingredients(data.get("ingredients", []))
    cuisine = data.get("cuisine", "Indian")
    diet = data.get("diet_goal", "Balanced")
    allergies = data.get("allergies", [])
    top_n = data.get("top_n", 1)

    if not ingredients:
        return jsonify({"error": "ingredients required"}), 400

    recipes = []
    for i in range(top_n):
        recipe = generate_recipe(ingredients=ingredients, cuisine=cuisine, diet=diet, allergies=allergies, variation=i)
        recipes.append(recipe)
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
    all_past_recipes = []
    for batch in search_history:
        if isinstance(batch, list):
            all_past_recipes.extend(batch)

    # Build ingredients query from profile
    ingredients = clean_ingredients(fav_ingredients) if fav_ingredients else ["vegetables"]
    target_cuisine = cuisines[0] if cuisines else "Indian"

    # If no history, generate 3 fresh recipes (survey mode)
    new_recipes = []
    if not all_past_recipes:
        for i in range(3):
            recipe = generate_recipe(
                ingredients=ingredients,
                cuisine=target_cuisine,
                diet=diet_goal or "Balanced",
                allergies=allergies,
                variation=i
            )
            new_recipes.append(recipe)

    # Return new + all past recipes
    return jsonify(new_recipes + all_past_recipes)


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

    user["favorites"] = [f for f in user["favorites"] if str(f.get("id")) != str(recipe_id)]
    save_db(db)
    return jsonify(user["favorites"])


# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
