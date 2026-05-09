import pandas as pd
import numpy as np
from pathlib import Path
import logging



# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]   # script/ → ALM/
META_PATH = ROOT / 'data' / 'processed' / 'metadata.csv'

# Create log dir first, THEN configure logging
log_dir = ROOT / 'outputs' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename = log_dir / 'add_text_description.log',
    filemode = 'a',
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    level    = logging.INFO
)
logger = logging.getLogger(__name__)
# ── Label → text description map ──────────────────────────────────────────
# Covers all 50 ESC-50 classes + 10 UrbanSound8K classes (58 unique after overlap)
DESCRIPTIONS = {
    # ── ESC-50 ──────────────────────────────────────────────────────────────
    "dog":                  ["a dog barking loudly",         "sound of a barking dog",           "aggressive dog bark"],
    "rooster":              ["a rooster crowing",            "cock-a-doodle-doo of a rooster",   "rooster call at dawn"],
    "pig":                  ["a pig oinking",                "sound of a pig grunting",          "pig squealing"],
    "cow":                  ["a cow mooing",                 "sound of cattle mooing",           "cow calling in a field"],
    "frog":                 ["a frog croaking",              "ribbit sound of a frog",           "frog calling at night"],
    "cat":                  ["a cat meowing",                "sound of a meowing cat",           "cat purring and meowing"],
    "hen":                  ["a hen clucking",               "chicken clucking sounds",          "hen making noise"],
    "insects":              ["insects buzzing and chirping", "sound of insects at night",        "cricket and insect sounds"],
    "sheep":                ["a sheep baaing",               "sound of a bleating sheep",        "sheep calling"],
    "crow":                 ["a crow cawing",                "sound of crows calling",           "crow making loud caw sounds"],
    "rain":                 ["sound of rain falling",        "raindrops hitting a surface",      "heavy rainfall sound"],
    "sea_waves":            ["ocean waves crashing",         "sound of sea waves on shore",      "waves rolling onto beach"],
    "crackling_fire":       ["fire crackling and popping",   "sound of a burning campfire",      "wood crackling in fire"],
    "crickets":             ["crickets chirping at night",   "sound of crickets in the grass",   "night cricket sounds"],
    "chirping_birds":       ["birds chirping and singing",   "sound of birds in the morning",    "bird calls in nature"],
    "water_drops":          ["water dripping sounds",        "droplets falling into water",      "sound of dripping water"],
    "wind":                 ["wind blowing loudly",          "sound of strong wind gusting",     "wind howling outdoors"],
    "pouring_water":        ["water being poured",           "sound of liquid pouring",          "pouring water into a glass"],
    "toilet_flush":         ["toilet flushing sound",        "sound of a flushing toilet",       "bathroom flush noise"],
    "thunderstorm":         ["thunder rumbling loudly",      "sound of a thunderstorm",          "lightning and thunder sounds"],
    "crying_baby":          ["a baby crying loudly",         "sound of an infant crying",        "baby wailing and crying"],
    "sneezing":             ["someone sneezing",             "loud sneeze sound",                "sneezing noise"],
    "clapping":             ["people clapping hands",        "applause and clapping sounds",     "hand clapping"],
    "breathing":            ["heavy breathing sounds",       "sound of deep breathing",          "person breathing audibly"],
    "coughing":             ["someone coughing repeatedly",  "cough sound",                      "dry coughing noise"],
    "footsteps":            ["footsteps on a hard floor",    "sound of someone walking",         "walking footstep sounds"],
    "laughing":             ["someone laughing out loud",    "laughter and giggling sounds",     "loud laughing"],
    "brushing_teeth":       ["sound of brushing teeth",      "toothbrush scrubbing sound",       "teeth brushing noise"],
    "snoring":              ["someone snoring loudly",       "snoring sleep sounds",             "loud snoring noise"],
    "drinking_sipping":     ["sound of someone drinking",   "sipping liquid sounds",            "drinking from a cup"],
    "door_wood_knock":      ["knocking on a wooden door",   "door knock sound",                 "someone knocking at door"],
    "mouse_click":          ["computer mouse clicking",      "sound of mouse button clicks",     "rapid mouse clicking"],
    "keyboard_typing":      ["typing on a keyboard",         "keyboard click sounds",            "rapid keyboard typing"],
    "door_wood_creaks":     ["wooden door creaking",         "sound of a creaky door",           "door hinge creaking"],
    "can_opening":          ["opening a metal can",          "sound of a can being opened",      "pop of a can opening"],
    "washing_machine":      ["washing machine running",      "sound of laundry machine spinning","washing machine noise"],
    "vacuum_cleaner":       ["vacuum cleaner running",       "sound of vacuuming floor",         "electric vacuum noise"],
    "clock_alarm":          ["alarm clock ringing",          "sound of a clock alarm",           "alarm buzzing loudly"],
    "clock_tick":           ["clock ticking steadily",       "sound of a ticking clock",         "tick tock of a clock"],
    "glass_breaking":       ["glass shattering loudly",      "sound of breaking glass",          "glass smashing"],
    "helicopter":           ["helicopter flying overhead",   "sound of a helicopter rotor",      "helicopter blades spinning"],
    "chainsaw":             ["chainsaw running loudly",      "sound of a chainsaw cutting",      "loud chainsaw motor"],
    "siren":                ["emergency siren wailing",      "loud siren alarm sound",           "police or ambulance siren"],
    "car_horn":             ["car horn honking",             "vehicle horn beeping",             "loud car horn sound"],
    "engine":               ["engine running and revving",   "sound of a motor engine",          "loud engine noise"],
    "train":                ["train passing by",             "sound of a train on tracks",       "railway train noise"],
    "church_bells":         ["church bells ringing",         "sound of bell chimes",             "tolling church bells"],
    "airplane":             ["airplane flying overhead",     "sound of a jet aircraft",          "plane engine roaring"],
    "fireworks":            ["fireworks exploding loudly",   "sound of fireworks bursting",      "firework pops and bangs"],
    "hand_saw":             ["hand saw cutting wood",        "sawing wood sound",                "manual saw scraping"],
    # ── UrbanSound8K-only ──────────────────────────────────────────────────
    "air_conditioner":      ["air conditioner humming",      "sound of AC unit running",         "HVAC system noise"],
    "children_playing":     ["children playing and shouting","kids playing outdoors",            "sound of children at play"],
    "drilling":             ["electric drill running",       "sound of drilling into wall",      "power drill noise"],
    "gun_shot":             ["gunshot firing loudly",        "sound of a gun being fired",       "loud gunfire bang"],
    "jackhammer":           ["jackhammer pounding pavement", "sound of a jackhammer",            "pneumatic drill hammering"],
    "street_music":         ["street musician playing",      "outdoor music performance",        "busker playing on street"],
    # ── Aliases / alternate spellings that may appear in CSV ───────────────
    "dog_bark":             ["a dog barking loudly",         "sound of a barking dog",           "aggressive dog bark"],
    "engine_idling":        ["car engine idling",            "sound of an idling motor",         "engine running at standstill"],
}

# ── Main ───────────────────────────────────────────────────────────────────
def add_descriptions(meta_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    df = pd.read_csv(meta_path)
    logger.info(f"Loaded  : {len(df)} rows from {meta_path}")
    logger.info(f"Labels  : {sorted(df['label'].unique())}")

    missing = [l for l in df['label'].unique() if l not in DESCRIPTIONS]
    if missing:
        logger.info(f"\n⚠  Labels not in description map — will use label name as fallback:")
        for m in missing:
            logger.info(f"   • {m}")

    def pick(label):
        opts = DESCRIPTIONS.get(label, [f"sound of {label.replace('_', ' ')}"])
        return opts[rng.integers(len(opts))]

    df['text_description'] = df['label'].apply(pick)

    # Reorder columns nicely
    cols = ['id', 'file', 'label', 'text_description', 'dataset', 'duration', 'fold', 'source_file']
    cols = [c for c in cols if c in df.columns]           # keep only existing cols
    df = df[cols]

    df.to_csv(meta_path, index=False)
    logger.info(f"\n✅ Saved with 'text_description' column → {meta_path}")
    logger.info(df[['label', 'text_description']].drop_duplicates().sort_values('label').to_string(index=False))


if __name__ == '__main__':
    add_descriptions(META_PATH)