"""
Inference script for the legal text decoder (HuBERT classifier).

Loads:
    - /app/model.pt (weights from 02-training.py)
    - data/final/test.csv (optional source for sample texts)

Outputs:
    - Predictions logged to stdout and saved to log/inference_outputs.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from utils import setup_logger

logger = setup_logger(__name__, "log/inference.log")

# Predefined unseen samples with expected labels
SAMPLE_TEXTS = [
    {"text": "Az MMM, illetve a Media Markt Webáruház nem vetette alá magát semmilyen magatartási kódex rendelkezéseinek.", "expected": 5},
    {"text": "A termékkel/szolgáltatással kapcsolatos fontos tudnivalókat (így különösen azok lényeges tulajdonságai, jellemzői; műszaki / technikai paraméterek stb.), valamint az eladási árakat a Vásárló a termék információs oldalán ismerheti meg még a megrendelése leadása előtt.", "expected": 4},
    {"text": "A Vásárlónak a rendelés bármely szakaszában és a megrendelés MMM részére való elküldéséig a Media Markt Webáruházban bármikor lehetősége van az adatbeviteli hibák javítására a megrendelési felületen. A részletes leírást az ÁSZF 5.1. pontja tartalmazza", "expected": 3},
    {"text": "TTávollévők között kötött szerződés esetén a vállalkozás a jogszabály által előírt tájékoztatást – világos és közérthető nyelven – a Vásárlóval közli, vagy azt a Vásárló számára elérhetővé teszi az alkalmazott távollévők közötti kommunikációt lehetővé tévő eszköznek megfelelő módon. A tartós adathordozón (ennek minősül az elektronikus levél, az e-mail is) rendelkezésre bocsátott tájékoztatásnak olvashatónak kell lennie. Ha a távollevők közötti, elektronikus úton kötött szerződés a Vásárló számára fizetési kötelezettséget keletkeztet, a vállalkozás egyértelműen és jól látható módon, közvetlenül a Vásárló szerződési nyilatkozatának megtétele előtt felhívja a Vásárló figyelmét a szerződés szerinti termék vagy szolgáltatás lényeges tulajdonságaira a szerződés szerinti termékért vagy szolgáltatásért járó ellenszolgáltatás adóval megnövelt teljes összegéről vagy – ha a termék vagy szolgáltatás jellegéből adódóan az ellenértéket nem lehet előre ésszerűen kiszámítani – annak számítási módjáról, valamint az ezen felül felmerülő valamennyi költségről (így különösen a fuvardíjról vagy a postaköltségről), vagy ha e költségeket nem lehet ésszerűen előre kiszámítani, annak a ténynek a feltüntetéséről, hogy további költségek merülhetnek fel, határozatlan időre szóló vagy előfizetést magában foglaló szerződés esetében arról, hogy az ellenszolgáltatás teljes összege a számlázási időszakra vonatkozó valamennyi költséget tartalmazza. Ha az ilyen szerződés átalánydíjas, arról, hogy az ellenszolgáltatás teljes összege egyúttal a teljes havi költséget is jelenti. Ha az összes költséget nem lehet előre kiszámítani, a Vásárlót tájékoztatni kell az ellenszolgáltatás összegének kiszámításának módjáról; határozott időre szóló szerződés esetén a szerződés időtartamáról, határozatlan időre szóló szerződés esetén a szerződés megszüntetésének feltételeiről; a határozott időre szóló olyan szerződés esetén, amely határozatlan időtartamúvá alakulhat át, az átalakulás feltételeiről, és az így határozatlan időtartamúvá átalakult szerződés megszüntetésének feltételeiről; a Vásárló kötelezettségeinek szerződés szerinti legrövidebb időtartamáról.", "expected": 1},
    {"text": "A kereskedelmi célú internetes honlappal rendelkező vállalkozás köteles legkésőbb a Vásárló szerződéses ajánlatának megtételekor egyértelműen és olvashatóan feltüntetni az esetleges fuvarozási korlátozásokat és az elfogadott fizetési módokat.", "expected": 5},
    {"text": "A távollévők között kötött szerződés megkötését követően – ésszerű időn belül, de a termék adásvételére irányuló szerződés esetén legkésőbb az átadáskor, a szolgáltatásnyújtására irányuló szerződés esetén legkésőbb a szolgáltatás teljesítésének megkezdésekor – a vállalkozás tartós adathordozón visszaigazolást ad a Vásárlónak a megkötött szerződésről. A visszaigazolás tartalmazza a fent részletesen meghatározott kötelező tájékoztatást, kivéve, ha azt a vállalkozás már a szerződés megkötése előtt tartós adathordozón a Vásárlónak megadta; és a Vásárló a 45/2014. (II.26) Korm. Rendelet 29. § m) pontja szerinti nyilatkozatot tett.", "expected": 3},
    {"text": "Annak érdekében, hogy az árfigyelő rendszer figyelmeztető üzenetet küldhessen, a szolgáltatást igénybe vevő látogató e-mail címének kezelésére van szükség. Amennyiben a látogató rendelkezik MediaMarkt Applikációval, és az ún. push üzeneteket engedélyezte, úgy az árfigyelő push értesítést is küld. Az adatkezelésről bővebb információ az Adatvédelmi Szabályzatunk 18/C. pontjában található.", "expected": 3},
    {"text": "6.2. A Vásárló részéről a megrendelés leadása nem eredményezi az MMM és a Vásárló közötti szerződés létrejöttét. A Vásárló által elküldött ajánlat (megrendelés) megérkezését az MMM késedelem nélkül, automatikusan generált e-mail útján, legkésőbb 48 órán belül visszaigazolja a Vásárló részére. Ez a visszaigazoló e-mail kizárólag arról tájékoztatja a Vásárlót, hogy a Media Markt Webáruházhoz a megrendelés megérkezett. Vásárló köteles a visszaigazoló e-mail tartalmát ellenőrizni, annak mellékletét ill. abban feltüntetett linkjeit részletesen áttekinteni, az általa megadott adatok, paraméterek helyességét leellenőrizni. A hibásan vagy nem kellő részletességgel megadott adatok, információk okán felmerülő bárminemű probléma, szállítási vagy egyéb többletköltség vagy ellehetetlenülés, késedelem Vásárlót terheli, az a Vásárló kizárólagos felelőssége.", "expected": 2},
    {"text": "Wolt Delivery Magyarország Kft. (székhely: 1085 Budapest, Salétrom utca 4; cégjegyzékszám: 01-09- 390993). A házhozszállító partner a szállítás teljesítéséhez vásárlói adatokat dolgoz fel. Az adatfeldolgozásról részletesen olvashat az MMM https://www.mediamarkt.hu/hu/legal/adatvedelem oldalon elérhető Adatvédelmi Szabályzatának 11. és 23.2. pontjában.", "expected": 3},
    {"text": "GLS General Logistics Systems Hungary Csomag-Logisztikai Kft. (székhely: 2351 Alsónémedi GLS Európa u. 2.; cégjegyzékszám: 13-09-111755), továbbá a Magyar Posta Zrt. (székhely: 1138 Budapest, Dunavirág u. 2-6, cégjegyzékszám: 01-10-042463). A csomagautomatát üzemeltető partner (a továbbiakban: üzemeltető partner) a szállítás teljesítéséhez vásárlói adatokat dolgoz fel. Az adatfeldolgozásról részletesen olvashat az MMM https://www.mediamarkt.hu/hu/shop/adatvedelem.html oldalon elérhető Adatvédelmi Szabályzatának 11. és 23.2. pontjában. A szolgáltatás igénybevételének feltétele, hogy a Vásárló a mobiltelefonszámát, valamint az e-mail címét megadja.", "expected": 2},
    {"text": "Amennyiben a raktár zárva tart, és a Vásárló a rendelését ezen a napon adja le, úgy a rendelés feldolgozása az első következő raktári munkanapon történik meg. A kiszállítás minden esetben arendelés feldolgozását követő munkanapon történik.Ennek megfelelően a péntek 15:00 óra és a következő hétfő 15:00 óra között beérkezett rendelések feldolgozását az MMM hétfőn vállalja, így ezen rendelések kiszállítására kedden kerül sor.", "expected": 3},
    {"text": "Az Utánvét díj összegéről azMMMa vásárlási folyamat során egyértelmű tájékoztatást ad a folyamat„Fizetés” elnevezésű szakaszában. Az Utánvét díj a szállítási költséghez adódik hozzá és az adott ügylet vonatkozásában kiállított számviteli bizonylaton jelenik meg szállítási költség/utánvét díj soron.", "expected": 4},
]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_model_path() -> Path:
    primary = Path(config.HF_MODEL_PATH)
    fallback = Path("/app/model.pt")
    if primary.exists():
        return primary
    if fallback.exists():
        logger.warning("Primary model path missing (%s); using fallback %s", primary, fallback)
        return fallback
    raise FileNotFoundError(f"Model weights not found at {primary} (or fallback {fallback}). Run 02-training.py after rebuilding the image.")


def _load_sample_texts() -> List[str]:
    test_csv = Path(config.TEST_CSV)
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        if "text" in df.columns and len(df):
            texts = df["text"].astype(str).str.strip().tolist()
            return texts[:5] + [s["text"] for s in SAMPLE_TEXTS]  # mix
    return [s["text"] for s in SAMPLE_TEXTS]


def _predict(texts: List[str]) -> List[Dict[str, object]]:
    device = _device()
    logger.info("Using device: %s", device)

    model_weights = _resolve_model_path()
    tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.HF_MODEL_NAME,
        num_labels=5,
    )
    state = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=config.HF_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        probs = outputs.logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

    results = []
    for text, pred, prob_vec in zip(texts, preds.cpu(), probs.cpu()):
        label = int(pred) + 1  # back to 1..5
        confidence = float(prob_vec[pred])
        results.append(
            {
                "text": text,
                "pred_label": label,
                "confidence": round(confidence, 4),
                "probs": [round(float(p), 4) for p in prob_vec],
            }
        )
    return results


def run_inference() -> None:
    texts = _load_sample_texts()
    if not texts:
        logger.warning("No texts available for inference.")
        return

    results = _predict(texts)

    # attach expected if provided
    expected_map = {s["text"]: s.get("expected") for s in SAMPLE_TEXTS}
    for r in results:
        r["expected"] = expected_map.get(r["text"])
        r["correct"] = (r["expected"] == r["pred_label"]) if r["expected"] is not None else None

    for r in results:
        snippet = r["text"][:120].replace("\n", " ")
        logger.info(
            "Pred: %s | conf=%.3f | expected=%s | correct=%s | text=%.120s",
            r["pred_label"],
            r["confidence"],
            r.get("expected"),
            r.get("correct"),
            snippet,
        )

    out_path = Path("log/inference_outputs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved inference outputs to %s", out_path)


if __name__ == "__main__":
    run_inference()
