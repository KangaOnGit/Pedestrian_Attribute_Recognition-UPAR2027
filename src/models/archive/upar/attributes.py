"""40 thuộc tính UPAR, cặp prompt khẳng định/phủ định (CoOp) và ràng buộc thuộc tính -> vùng cơ thể."""

COLORS = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"]
ATTR_NAMES = (
    ["Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
     "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short"]
    + ["UpperBody-Color-" + c for c in COLORS]
    + ["LowerBody-Length-Short"]
    + ["LowerBody-Color-" + c for c in COLORS]
    + ["LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
       "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"]
)
NUM_ATTR = len(ATTR_NAMES)
assert NUM_ATTR == 40
assert ATTR_NAMES[8] == "UpperBody-Color-Black" and ATTR_NAMES[19] == "UpperBody-Color-Other"
assert ATTR_NAMES[20] == "LowerBody-Length-Short" and ATTR_NAMES[21] == "LowerBody-Color-Black"
assert ATTR_NAMES[32] == "LowerBody-Color-Other" and ATTR_NAMES[33] == "LowerBody-Type-Trousers&Shorts"
assert ATTR_NAMES[35] == "Accessory-Backpack" and ATTR_NAMES[39] == "Accessory-Hat"


def build_phrases():
    """Cụm mô tả (khẳng định, phủ định) cho từng thuộc tính; prompt = '[V1..Vn] <cụm>.'"""
    P = {
        "Age-Young": ("young person", "person who is not young"),
        "Age-Adult": ("adult person", "person who is not an adult"),
        "Age-Old": ("elderly person", "person who is not elderly"),
        "Gender-Female": ("woman", "man"),
        "Hair-Length-Short": ("person with short hair", "person without short hair"),
        "Hair-Length-Long": ("person with long hair", "person without long hair"),
        "Hair-Length-Bald": ("bald person", "person with hair"),
        "UpperBody-Length-Short": ("person wearing a short-sleeved top", "person wearing a long-sleeved top"),
        "LowerBody-Length-Short": ("person wearing shorts", "person wearing long trousers"),
        "LowerBody-Type-Trousers&Shorts": ("person wearing trousers or shorts", "person not wearing trousers or shorts"),
        "LowerBody-Type-Skirt&Dress": ("person wearing a skirt or a dress", "person not wearing a skirt or a dress"),
        "Accessory-Backpack": ("person carrying a backpack", "person without a backpack"),
        "Accessory-Bag": ("person carrying a bag", "person carrying no bag"),
        "Accessory-Glasses-Normal": ("person wearing glasses", "person wearing no glasses"),
        "Accessory-Glasses-Sun": ("person wearing sunglasses", "person wearing no sunglasses"),
        "Accessory-Hat": ("person wearing a hat", "person wearing no hat"),
    }
    for c in COLORS:
        cl = c.lower()
        if c == "Other":
            P["UpperBody-Color-" + c] = ("person wearing a top of another color", "person wearing a top of a basic color")
            P["LowerBody-Color-" + c] = ("person wearing trousers of another color", "person wearing trousers of a basic color")
        else:
            P["UpperBody-Color-" + c] = ("person wearing a %s top" % cl, "person not wearing a %s top" % cl)
            P["LowerBody-Color-" + c] = ("person wearing %s trousers" % cl, "person not wearing %s trousers" % cl)
    return P


PHRASES = build_phrases()
assert set(PHRASES) == set(ATTR_NAMES)
# (phủ định, khẳng định): chỉ số 0 = phủ định, 1 = khẳng định
PROMPT_PAIRS = [(PHRASES[n][1], PHRASES[n][0]) for n in ATTR_NAMES]

# ---- ràng buộc không gian: thuộc tính nào chỉ được so với vùng nào ----
PART_HEAD, PART_BODY, PART_LEGS, PART_GLOBAL = 0, 1, 2, 3
PART_TITLES = {PART_HEAD: "Đầu (Top)", PART_BODY: "Thân (Middle)", PART_LEGS: "Chân (Bottom)",
               PART_GLOBAL: "Toàn thân (mean 3 dải)"}


def part_of(name):
    if name.startswith(("Age", "Gender")):
        return PART_GLOBAL
    if name.startswith(("Hair", "Accessory-Glasses", "Accessory-Hat")):
        return PART_HEAD
    if name.startswith(("UpperBody", "Accessory-Backpack", "Accessory-Bag")):
        return PART_BODY
    if name.startswith("LowerBody"):
        return PART_LEGS
    raise ValueError(name)


PART_OF_ATTR = [part_of(n) for n in ATTR_NAMES]
