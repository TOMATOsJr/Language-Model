"""Hindi lexical resources for the synthetic comparison grammar."""

from __future__ import annotations

HINDI_PERSON_NAMES = [
    "आरव", "रोहन", "अमन", "सिया", "काव्या", "विवान", "अर्जुन", "अनन्या",
    "दीपक", "निशा", "आदित्य", "तृप्ति", "राहुल", "पूजा", "नेहा", "विकास",
    "सोनिया", "करण", "मीरा", "अजय",
]
HINDI_OBJECTS = [
    "किताब", "बैग", "बक्सा", "कुर्सी", "टेबल", "बोतल", "कार", "साइकिल",
    "मोबाइल", "कंप्यूटर", "घड़ी", "खिलौना",
]
HINDI_PLACES = ["दिल्ली", "मुंबई", "जयपुर", "आगरा", "लखनऊ", "पटना", "भोपाल", "पुणे"]
HINDI_ANIMALS = ["हाथी", "घोड़ा", "कुत्ता", "बिल्ली", "गाय", "भैंस", "बकरी", "शेर", "हिरण", "ऊंट"]

HINDI_NAMES = HINDI_PERSON_NAMES

ENTITY_POOLS = {
    "person": HINDI_PERSON_NAMES,
    "object": HINDI_OBJECTS,
    "place": HINDI_PLACES,
    "animal": HINDI_ANIMALS,
}

DOMAIN_LEXICON = {
    "height": {"measure": "ऊंचाई", "greater_adjective": "लंबा", "less_adjective": "छोटा", "categories": ("person", "object", "animal")},
    "age": {"measure": "उम्र", "greater_adjective": "बड़ा", "less_adjective": "छोटा", "categories": ("person", "animal")},
    "price": {"measure": "कीमत", "greater_adjective": "महंगा", "less_adjective": "सस्ता", "categories": ("object",)},
    "weight": {"measure": "वजन", "greater_adjective": "भारी", "less_adjective": "हल्का", "categories": ("object", "animal")},
    "quantity": {"measure": "मात्रा", "greater_adjective": "अधिक", "less_adjective": "कम", "categories": ("object",)},
    "distance": {"measure": "दूरी", "greater_adjective": "दूर", "less_adjective": "पास", "categories": ("place",)},
    "speed": {"measure": "गति", "greater_adjective": "तेज", "less_adjective": "धीमा", "categories": ("object", "animal")},
    "length": {"measure": "लंबाई", "greater_adjective": "लंबा", "less_adjective": "छोटा", "categories": ("object",)},
    "size": {"measure": "आकार", "greater_adjective": "बड़ा", "less_adjective": "छोटा", "categories": ("object", "animal")},
    "temperature": {"measure": "तापमान", "greater_adjective": "गर्म", "less_adjective": "ठंडा", "categories": ("place", "object")},
    "duration": {"measure": "अवधि", "greater_adjective": "लंबा", "less_adjective": "छोटा", "categories": ("object",)},
    "score": {"measure": "अंक", "greater_adjective": "अधिक", "less_adjective": "कम", "categories": ("person",)},
}

DOMAIN_WORDS = DOMAIN_LEXICON
