"""Nepali lexical resources for the synthetic comparison grammar."""

from __future__ import annotations

NEPALI_PERSON_NAMES = [
    "आरव", "रोहन", "अमन", "सिया", "काव्या", "विवान", "अर्जुन", "अनन्या",
    "दीपक", "निशा", "आदित्य", "तृप्ति", "राहुल", "पूजा", "नेहा", "विकास",
    "सोनिया", "करण", "मीरा", "अजय",
]
NEPALI_OBJECTS = ["किताब", "झोला", "बाकस", "कुर्सी", "टेबल", "बोतल", "कार", "साइकल", "मोबाइल", "कम्प्युटर", "घडी", "खेलौना"]
NEPALI_PLACES = ["काठमाडौं", "पोखरा", "ललितपुर", "भक्तपुर", "चितवन", "विराटनगर", "जनकपुर", "बुटवल"]
NEPALI_ANIMALS = ["हात्ती", "घोडा", "कुकुर", "बिरालो", "गाई", "भैंसी", "बाख्रा", "सिंह", "मृग", "ऊँट"]

NEPALI_NAMES = NEPALI_PERSON_NAMES

ENTITY_POOLS = {
    "person": NEPALI_PERSON_NAMES,
    "object": NEPALI_OBJECTS,
    "place": NEPALI_PLACES,
    "animal": NEPALI_ANIMALS,
}

DOMAIN_LEXICON = {
    "height": {"measure": "उचाइ", "greater_adjective": "अग्लो", "less_adjective": "होचो", "categories": ("person", "object", "animal")},
    "age": {"measure": "उमेर", "greater_adjective": "ठूलो", "less_adjective": "सानो", "categories": ("person", "animal")},
    "price": {"measure": "मूल्य", "greater_adjective": "महँगो", "less_adjective": "सस्तो", "categories": ("object",)},
    "weight": {"measure": "तौल", "greater_adjective": "भारी", "less_adjective": "हल्का", "categories": ("object", "animal")},
    "quantity": {"measure": "मात्रा", "greater_adjective": "धेरै", "less_adjective": "थोरै", "categories": ("object",)},
    "distance": {"measure": "दूरी", "greater_adjective": "टाढा", "less_adjective": "नजिक", "categories": ("place",)},
    "speed": {"measure": "गति", "greater_adjective": "छिटो", "less_adjective": "ढिलो", "categories": ("object", "animal")},
    "length": {"measure": "लम्बाइ", "greater_adjective": "लामो", "less_adjective": "छोटो", "categories": ("object",)},
    "size": {"measure": "आकार", "greater_adjective": "ठूलो", "less_adjective": "सानो", "categories": ("object", "animal")},
    "temperature": {"measure": "तापक्रम", "greater_adjective": "तातो", "less_adjective": "चिसो", "categories": ("place", "object")},
    "duration": {"measure": "अवधि", "greater_adjective": "लामो", "less_adjective": "छोटो", "categories": ("object",)},
    "score": {"measure": "अंक", "greater_adjective": "धेरै", "less_adjective": "कम", "categories": ("person",)},
}

DOMAIN_WORDS = DOMAIN_LEXICON
