# Arabic letter transliterated names → Unicode isolated form
ARABIC_LETTERS: dict[str, str] = {
    "alef": "\u0627", "alif": "\u0627",
    "ba": "\u0628", "baa": "\u0628",
    "ta": "\u062a", "taa": "\u062a",
    "tha": "\u062b", "thaa": "\u062b",
    "jim": "\u062c", "jeem": "\u062c",
    "ha": "\u062d", "haa": "\u062d",
    "kha": "\u062e", "khaa": "\u062e",
    "dal": "\u062f", "daal": "\u062f",
    "dhal": "\u0630", "dhaal": "\u0630", "thal": "\u0630",
    "ra": "\u0631", "raa": "\u0631",
    "zay": "\u0632", "zayn": "\u0632", "zain": "\u0632",
    "sin": "\u0633", "seen": "\u0633",
    "shin": "\u0634", "sheen": "\u0634",
    "sad": "\u0635", "saad": "\u0635",
    "dad": "\u0636", "daad": "\u0636",
    "tah": "\u0637",
    "dha": "\u0638", "dhaa": "\u0638",
    "ain": "\u0639", "ayn": "\u0639",
    "ghain": "\u063a", "ghayn": "\u063a",
    "fa": "\u0641", "faa": "\u0641",
    "qaf": "\u0642", "qaaf": "\u0642",
    "kaf": "\u0643", "kaaf": "\u0643",
    "lam": "\u0644",
    "mim": "\u0645", "meem": "\u0645",
    "nun": "\u0646", "noon": "\u0646",
    "heh": "\u0647",
    "waw": "\u0648",
    "ya": "\u064a", "yaa": "\u064a",
}


def resolve_arabic_letter(label: str) -> str | None:
    """Return the Arabic Unicode character for a label.

    Accepts either:
    - A single Arabic Unicode character directly (e.g. 'ب')
    - A transliterated name (e.g. 'ba', 'BA', 'lam')

    Returns None if the label cannot be resolved.
    """
    label = label.strip()
    if not label:
        return None
    # Direct Arabic Unicode character (U+0600–U+06FF Arabic block)
    if len(label) == 1 and "\u0600" <= label <= "\u06ff":
        return label
    return ARABIC_LETTERS.get(label.lower())


def join_arabic_letters(letters: list[str]) -> str:
    """Concatenate Arabic Unicode characters into a word.

    Unicode + the browser's Arabic-capable font handle contextual shaping
    (isolated / initial / medial / final forms) automatically.
    Letters should be provided in right-to-left logical order (first letter
    of the word first in the list).
    """
    return "".join(letters)
