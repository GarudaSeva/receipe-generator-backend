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
# Remove any None or empty keys
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise RuntimeError(
        "No Gemini API keys found. Set at least GEMINI_API_KEY_1 in your .env file."
    )

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

    # Expand each item by splitting on commas/spaces too
    # (handles cases like ["egg,rice,onion"] sent as a single list element)
    expanded = []
    for i in ingredients:
        parts = re.split(r"[,\s]+", str(i))
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

DISH_STYLES = ["curry", "fry", "gravy", "masala", "biryani"]

def build_prompt(ingredients, cuisine="Indian", diet="Balanced", allergies=None, variation=0):

    allergies = allergies or []
    style_hint = DISH_STYLES[variation % len(DISH_STYLES)]

    return f"""You are an expert Indian home chef and nutritionist.

Ingredients available: {", ".join(ingredients)}
Cuisine: {cuisine}
Diet: {diet}
Allergies to avoid: {", ".join(allergies) if allergies else "None"}
Style: {style_hint}

TASK: Create one authentic, detailed Indian {style_hint} recipe using the ingredients above.

MANDATORY RULES:
1. Choose a REAL dish name (e.g. "Chicken Curry", "Egg Fried Rice", "Aloo Masala").
2. Include ALL common Indian kitchen staples with exact quantities: cooking oil, onions, garlic, ginger, green chillies, curry leaves, mustard seeds, cumin seeds, turmeric powder, red chilli powder, coriander powder, garam masala, salt, fresh coriander.
3. Every process step MUST be specific with actions, timings and quantities. Include at least 10 detailed steps.
4. Respect allergies strictly. Match the diet type: {diet}.
5. Nutrition MUST list per-serving values for: Calories, Protein, Carbohydrates, Fat, Fiber, Sodium.
6. Benefits MUST be 4-5 detailed sentences explaining specific health benefits of the main ingredients and spices.
7. Include at least 4 practical chef tips specific to this dish.

OUTPUT: Return ONLY a raw JSON object. No markdown, no backticks, no explanation before or after the JSON.
All array items MUST be plain strings — NOT objects or numbered keys.

Exact JSON structure to follow:

{{
  "name": "Chicken Curry",
  "cuisine": "{cuisine}",
  "dietType": "{diet}",
  "ingredients": [
    "500g chicken, cut into 2-inch pieces",
    "2 medium onions, finely chopped",
    "2 ripe tomatoes, pureed",
    "2 tbsp cooking oil",
    "1 tsp cumin seeds",
    "1 tbsp ginger-garlic paste",
    "8-10 curry leaves",
    "2 green chillies, slit",
    "1 tsp turmeric powder",
    "1.5 tsp red chilli powder",
    "1 tsp coriander powder",
    "0.5 tsp garam masala",
    "salt to taste",
    "2 tbsp fresh coriander, chopped"
  ],
  "prepTime": "15 minutes",
  "cookTime": "35 minutes",
  "servings": "3-4 people",
  "difficulty": "Medium",
  "process": [
    "Wash the chicken pieces thoroughly under cold water, pat completely dry with paper towels and set aside.",
    "Heat 2 tbsp oil in a heavy-bottomed kadai over medium-high heat until shimmering hot.",
    "Add 1 tsp cumin seeds and wait 20-30 seconds until they splutter and turn golden brown.",
    "Add curry leaves and slit green chillies, fry for 15 seconds until fragrant.",
    "Add finely chopped onions and cook on medium heat for 8-10 minutes, stirring often, until deep golden brown.",
    "Add 1 tbsp ginger-garlic paste and cook for 2 minutes, stirring constantly, until the raw smell disappears.",
    "Add pureed tomatoes and cook for 5-6 minutes until the oil starts to separate from the masala.",
    "Lower the heat and add turmeric, red chilli powder and coriander powder. Stir for 30 seconds.",
    "Add the chicken pieces, increase heat to medium-high and sear for 4-5 minutes, turning to brown all sides.",
    "Add 3/4 cup warm water and salt to taste. Stir well to coat chicken evenly with the masala.",
    "Cover with a tight lid, reduce heat to low and cook for 20 minutes. Stir once halfway through.",
    "Uncover, increase heat to medium and cook for 3-4 more minutes to reduce gravy to desired consistency.",
    "Sprinkle garam masala, stir gently and cook for 1 final minute. Turn off the heat.",
    "Garnish generously with freshly chopped coriander leaves. Serve hot with steamed rice or chapati."
  ],
  "nutrition": "Calories: ~420 kcal\\nProtein: 32g\\nCarbohydrates: 18g\\nFat: 22g\\nFiber: 3g\\nSodium: 580mg",
  "benefits": "Chicken is an excellent source of lean protein that supports muscle repair, growth and immune function. Turmeric contains curcumin, a powerful anti-inflammatory compound that supports joint health, brain function and immunity. Ginger and garlic have proven antibacterial and antioxidant properties that aid digestion and support cardiovascular health. Cumin seeds stimulate digestive enzymes and help reduce bloating and acidity after meals. The combination of these spices provides essential micronutrients including iron, magnesium, zinc and B-vitamins that support energy metabolism.",
  "tips": [
    "Pat chicken completely dry before cooking — any surface moisture prevents proper searing and results in a watery curry.",
    "Fry the onions with patience on medium heat — deeply caramelized golden-brown onions are the biggest secret to a rich, full-bodied curry.",
    "Always add dry spice powders on low heat and stir immediately for 30 seconds — high heat burns the spices and turns the dish bitter.",
    "For a richer, creamier gravy, stir in 2 tbsp fresh cream or thick coconut milk in the final 2 minutes of cooking.",
    "Rest the curry for 5-10 minutes off the heat before serving — the flavors continue to develop and deepen as it sits."
  ],
  "pairing": "Serve with steamed basmati rice or butter naan for a complete meal. A side of sliced onions drizzled with lemon juice, fresh cucumber raita and a wedge of lemon complement the spices beautifully. Finish the meal with a tall glass of chilled salted lassi to balance the heat."
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
    print(f"[DEBUG] Generating recipe #{variation+1} for: {ingredients}")

    for attempt in range(2):
        try:
            if attempt > 0:
                print(f"[DEBUG] Retry attempt 2 with next API key, waiting 2 seconds...")
                time.sleep(2)

            current_model = get_next_model()
            response = current_model.generate_content(prompt)
            text = response.text.strip()
            print(f"[DEBUG] AI Response received, length: {len(text)}")

            # Strip markdown code fences Gemini sometimes wraps responses in
            if "```" in text:
                fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
                if fence_match:
                    text = fence_match.group(1).strip()

            # Extract JSON object from the response
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                print(f"[DEBUG] No JSON object found in response: {text[:300]}")
                raise Exception("No JSON object found in AI response")

            data = json.loads(match.group(0))

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

            return {
                "id": str(uuid.uuid4()),
                "name": str(data.get("name", "")).strip(),
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
            print(f"[DEBUG] AI Error (attempt {attempt+1}): {type(e).__name__}: {error_msg}")
            if attempt == 0:
                # Always retry with the next API key on ANY first-attempt failure
                print(f"[DEBUG] First attempt failed ({type(e).__name__}), retrying with next key...")
                continue
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] Both attempts failed, using fallback recipe")
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
        "1 medium onion, finely chopped",
        "1 tbsp ginger-garlic paste",
        "8-10 curry leaves",
        "2 green chillies, slit",
        "1 tsp mustard seeds",
        "1 tsp cumin seeds",
        "1/2 tsp turmeric powder",
        "1 tsp red chilli powder",
        "1 tsp coriander powder",
        "1/2 tsp garam masala",
        "salt to taste",
        "2 tbsp fresh coriander, chopped for garnish"
    ]

    # Detect diet type from the main ingredient
    meat_keywords = ["chicken", "mutton", "fish", "prawn", "shrimp", "beef", "pork", "lamb"]
    egg_keywords = ["egg", "eggs"]
    if any(k in main.lower() for k in meat_keywords):
        diet_type = "Non Vegetarian"
    elif any(k in main.lower() for k in egg_keywords):
        diet_type = "Non Vegetarian"
    else:
        diet_type = "Vegetarian"

    style = dish["suffix"]

    return {
        "id": str(uuid.uuid4()),
        "name": f"{main.title()} {style}",
        "cuisine": "Indian",
        "dietType": diet_type,
        "ingredients": full_ingredients,
        "prepTime": "15 minutes",
        "cookTime": "25 minutes",
        "servings": "2-3 people",
        "difficulty": "Easy",
        "process": process,
        "nutrition": "Calories: ~280 kcal\nProtein: 8g\nCarbohydrates: 32g\nFat: 12g\nFiber: 5g\nSodium: 490mg",
        "benefits": (
            "Turmeric contains curcumin, a powerful anti-inflammatory compound that supports joint health and immunity. "
            "Mustard seeds and curry leaves are rich in antioxidants that protect cells from oxidative stress. "
            "Ginger aids digestion, reduces nausea and has proven anti-bacterial properties that support gut health. "
            "Cumin seeds stimulate digestive enzymes, helping reduce bloating and improving nutrient absorption. "
            "The combination of these spices provides iron, magnesium and essential B-vitamins that support energy metabolism and immune function."
        ),
        "tips": [
            "Always heat the oil well before adding mustard seeds — cold oil prevents the seeds from spluttering properly.",
            "Add powdered spices on low heat and stir immediately for 30 seconds — high heat burns them and makes the dish bitter.",
            "Use fresh curry leaves for maximum fragrance — dried ones lose most of their aroma and flavor.",
            "Taste and adjust salt only at the end of cooking, as the flavors concentrate as the dish reduces.",
            f"For best texture, do not cut {main} pieces too small — medium-sized pieces hold their shape and absorb the masala well."
        ],
        "pairing": f"{main.title()} {style} pairs beautifully with steamed basmati rice or freshly made chapati. Serve alongside a simple cucumber and onion salad dressed with lemon juice and a pinch of chaat masala. A small bowl of plain yogurt or raita on the side helps balance the spice levels.",
        "calories": 280
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
