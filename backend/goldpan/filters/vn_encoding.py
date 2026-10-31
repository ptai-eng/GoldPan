"""Vietnamese legacy encoding detector/converter (TCVN3, VNI-Windows, NFD -> NFC) and diacritic folder."""

import re
import unicodedata

# VNI-Windows vowel + tone/modifier sequences.
# Strictly ordered longest-first (3-char -> 2-char -> 1-char) so single-byte tone markers
# (like ï, ø, ù, û, õ, â, ê, ô, ö) never clobber multi-char sequences (like hôïp -> hợp).
_VNI_SEQUENCES: list[tuple[str, str]] = [
    # 3-character sequences (ư + ơ compound öô / ÖÔ with tones)
    ("öôø", "ườ"),
    ("öôù", "ướ"),
    ("öôû", "ưở"),
    ("öôõ", "ưỡ"),
    ("öôï", "ượ"),
    ("ÖÔØ", "ƯỜ"),
    ("ÖÔÙ", "ƯỚ"),
    ("ÖÔÛ", "ƯỞ"),
    ("ÖÔÕ", "ƯỠ"),
    ("ÖÔÏ", "ƯỢ"),
    # 2-character sequences: a / â / ă
    ("aø", "à"),
    ("aù", "á"),
    ("aû", "ả"),
    ("aõ", "ã"),
    ("aï", "ạ"),
    ("aà", "ầ"),
    ("aá", "ấ"),
    ("aå", "ẩ"),
    ("aã", "ẫ"),
    ("aä", "ậ"),
    ("aâ", "â"),
    ("aè", "ằ"),
    ("aé", "ắ"),
    ("aë", "ẳ"),
    ("aẵ", "ẵ"),
    ("aặ", "ặ"),
    ("aê", "ă"),
    ("AØ", "À"),
    ("AÙ", "Á"),
    ("AÛ", "Ả"),
    ("AÕ", "Ã"),
    ("AÏ", "Ạ"),
    ("AÀ", "Ầ"),
    ("AÁ", "Ấ"),
    ("AÅ", "Ẩ"),
    ("AÃ", "Ẫ"),
    ("AÄ", "Ậ"),
    ("AÂ", "Â"),
    ("AÈ", "Ằ"),
    ("AÉ", "Ắ"),
    ("AË", "Ẳ"),
    ("AÊ", "Ă"),
    # 2-character sequences: e / ê
    ("eø", "è"),
    ("eù", "é"),
    ("eû", "ẻ"),
    ("eõ", "ẽ"),
    ("eï", "ẹ"),
    ("eà", "ề"),
    ("eá", "ế"),
    ("eå", "ể"),
    ("eã", "ễ"),
    ("eä", "ệ"),
    ("eâ", "ê"),
    ("EØ", "È"),
    ("EÙ", "É"),
    ("EÛ", "Ẻ"),
    ("EÕ", "Ẽ"),
    ("EÏ", "Ẹ"),
    ("EÀ", "Ề"),
    ("EÁ", "Ế"),
    ("EÅ", "Ể"),
    ("EÃ", "Ễ"),
    ("EÄ", "Ệ"),
    ("EÂ", "Ê"),
    # 2-character sequences: o / ô / ơ / ươ
    ("öô", "ươ"),
    ("ÖÔ", "ƯƠ"),
    ("oø", "ò"),
    ("où", "ó"),
    ("oû", "ỏ"),
    ("oõ", "õ"),
    ("oï", "ọ"),
    ("oà", "ồ"),
    ("oá", "ố"),
    ("oå", "ổ"),
    ("oã", "ỗ"),
    ("oä", "ộ"),
    ("oâ", "ô"),
    ("ôø", "ờ"),
    ("ôù", "ớ"),
    ("ôû", "ở"),
    ("ôõ", "ỡ"),
    ("ôï", "ợ"),
    ("OØ", "Ò"),
    ("OÙ", "Ó"),
    ("OÛ", "Ỏ"),
    ("OÕ", "Õ"),
    ("OÏ", "Ọ"),
    ("OÀ", "Ồ"),
    ("OÁ", "Ố"),
    ("OÅ", "Ổ"),
    ("OÃ", "Ỗ"),
    ("OÄ", "Ộ"),
    ("OÂ", "Ô"),
    ("ÔØ", "Ờ"),
    ("ÔÙ", "Ớ"),
    ("ÔÛ", "Ở"),
    ("ÔÕ", "Ỡ"),
    ("ÔÏ", "Ợ"),
    # 2-character sequences: u / ư
    ("uø", "ù"),
    ("uù", "ú"),
    ("uû", "ủ"),
    ("uõ", "ũ"),
    ("uï", "ụ"),
    ("öø", "ừ"),
    ("öù", "ứ"),
    ("öû", "ử"),
    ("öõ", "ữ"),
    ("öï", "ự"),
    ("UØ", "Ù"),
    ("UÙ", "Ú"),
    ("UÛ", "Ủ"),
    ("UÕ", "Ũ"),
    ("UÏ", "Ụ"),
    ("ÖØ", "Ừ"),
    ("ÖÙ", "Ứ"),
    ("ÖÛ", "Ử"),
    ("ÖÕ", "Ữ"),
    ("ÖÏ", "Ự"),
    # 2-character sequences: y
    ("yø", "ỳ"),
    ("yù", "ý"),
    ("yû", "ỷ"),
    ("yõ", "ỹ"),
    ("YØ", "Ỳ"),
    ("YÙ", "Ý"),
    ("YÛ", "Ỷ"),
    ("YÕ", "Ỹ"),
    # 1-character VNI vowel/consonant mappings (applied only after all 2/3-char sequences)
    ("ñ", "đ"),
    ("Ñ", "Đ"),
    ("ô", "ơ"),
    ("Ô", "Ơ"),
    ("ö", "ư"),
    ("Ö", "Ư"),
    ("ò", "ị"),
    ("Ò", "Ị"),
]

# TCVN3 (ABC) multi-char sequences and single-char table (longest-first)
_TCVN3_SEQUENCES: list[tuple[str, str]] = [
    ("®­îc", "được"),
    ("­î", "ượ"),
    ("­í", "ướ"),
    ("­ì", "ườ"),
    ("­ë", "ưở"),
    ("­¬", "ươ"),
    ("¸", "á"),
    ("µ", "à"),
    ("¶", "ả"),
    ("·", "ã"),
    ("¹", "ạ"),
    ("¨", "ă"),
    ("»", "ằ"),
    ("¾", "ắ"),
    ("¼", "ẳ"),
    ("½", "ẵ"),
    ("Æ", "ặ"),
    ("©", "â"),
    ("Ç", "ầ"),
    ("Ê", "ấ"),
    ("È", "ẩ"),
    ("É", "ẫ"),
    ("Ë", "ậ"),
    ("Ð", "é"),
    ("Ì", "è"),
    ("Î", "ẻ"),
    ("Ï", "ẽ"),
    ("Ñ", "ẹ"),
    ("ª", "ê"),
    ("Ò", "ề"),
    ("Õ", "ế"),
    ("Ó", "ể"),
    ("Ô", "ễ"),
    ("Ö", "ệ"),
    ("Ý", "í"),
    ("×", "ì"),
    ("Ø", "ỉ"),
    ("Ü", "ĩ"),
    ("Þ", "ị"),
    ("ã", "ó"),
    ("ß", "ò"),
    ("á", "ỏ"),
    ("â", "õ"),
    ("ä", "ọ"),
    ("«", "ô"),
    ("å", "ồ"),
    ("è", "ố"),
    ("æ", "ổ"),
    ("ç", "ỗ"),
    ("é", "ộ"),
    ("¬", "ơ"),
    ("ê", "ờ"),
    ("í", "ớ"),
    ("ë", "ở"),
    ("ì", "ỡ"),
    ("î", "ợ"),
    ("ó", "ú"),
    ("ï", "ù"),
    ("ñ", "ủ"),
    ("ò", "ũ"),
    ("ô", "ụ"),
    ("­", "ư"),
    ("õ", "ừ"),
    ("ø", "ứ"),
    ("ö", "ử"),
    ("÷", "ữ"),
    ("ù", "ự"),
    ("ý", "ý"),
    ("ú", "ỳ"),
    ("û", "ỷ"),
    ("ü", "ỹ"),
    ("þ", "ỵ"),
    ("®", "đ"),
    ("¡", "Ă"),
    ("¢", "Â"),
    ("£", "Ê"),
    ("¤", "Ô"),
    ("¥", "Ơ"),
    ("¦", "Ư"),
    ("§", "Đ"),
]

_VNI_SIGNATURE_RE = re.compile(
    r"(?:aø|aù|aû|aõ|aï|aâ|aà|aá|aå|aä|eø|eù|eâ|eà|eá|eä|oø|où|oâ|oà|oá|oä|ôø|ôù|ôû|ôï|uø|uù|uû|uï|öô|öù|öø|öû|öï|ñ|Ñ)"
)
_TCVN3_SIGNATURE_RE = re.compile(
    r"(?:®­îc|®ång|nghiÖp|xuÊt|khÈu|thuÕ|gi¸|trÞ|t¨ng|ch­a|khÊu|trõ|hå\s+s¬|hoµn|th¸ng|hoÆc|[¸µ¶·¹¨»¾¼½Æ©ÇÊÈÉËÒÕÓÔÖ®­¬]{2,})"
)


def detect_vietnamese_encoding(text: str) -> str:
    """Detect whether text contains VNI-Windows, TCVN3 (ABC), Unicode NFD, or standard NFC."""
    if not text:
        return "unicode_nfc"

    vni_matches = len(_VNI_SIGNATURE_RE.findall(text))
    tcvn3_matches = len(_TCVN3_SIGNATURE_RE.findall(text))

    if vni_matches >= 2 and vni_matches >= tcvn3_matches:
        return "vni_windows"
    if tcvn3_matches >= 2:
        return "tcvn3_abc"

    if unicodedata.normalize("NFC", text) != text:
        return "unicode_nfd"
    return "unicode_nfc"


_VNI_MAP: dict[str, str] = dict(_VNI_SEQUENCES)
_VNI_TOKEN_RE = re.compile(
    "|".join(re.escape(k) for k, _ in sorted(_VNI_SEQUENCES, key=lambda x: len(x[0]), reverse=True))
)

_TCVN3_MAP: dict[str, str] = dict(_TCVN3_SEQUENCES)
_TCVN3_TOKEN_RE = re.compile(
    "|".join(re.escape(k) for k, _ in sorted(_TCVN3_SEQUENCES, key=lambda x: len(x[0]), reverse=True))
)


def _convert_vni_word(word: str) -> str:
    """Convert a single VNI-encoded token into Unicode NFC in a single non-overlapping pass."""
    return _VNI_TOKEN_RE.sub(lambda m: _VNI_MAP[m.group(0)], word)


def _convert_tcvn3_word(word: str) -> str:
    """Convert a single TCVN3-encoded token into Unicode NFC in a single non-overlapping pass."""
    return _TCVN3_TOKEN_RE.sub(lambda m: _TCVN3_MAP[m.group(0)], word)


def normalize_vietnamese_text(text: str) -> tuple[str, str]:
    """Normalize Vietnamese text from VNI-Windows, TCVN3, or Unicode NFD into clean Unicode NFC."""
    if not text:
        return "", "unicode_nfc"

    encoding = detect_vietnamese_encoding(text)

    if encoding == "vni_windows":
        tokens = re.split(r"(\s+)", text)
        converted: list[str] = []
        for tok in tokens:
            if _VNI_SIGNATURE_RE.search(tok) or any(ch in tok for ch in ("ñ", "Ñ", "ö", "Ö", "ô", "Ô")):
                converted.append(_convert_vni_word(tok))
            else:
                converted.append(tok)
        result = "".join(converted)
        return unicodedata.normalize("NFC", result), encoding

    if encoding == "tcvn3_abc":
        tokens = re.split(r"(\s+)", text)
        converted = []
        for tok in tokens:
            if _TCVN3_SIGNATURE_RE.search(tok) or any(
                ch in tok for ch in "¸µ¶·¹¨»¾¼½Æ©ÇÊÈÉËÒÕÓÔÖ®­¬Þßãäåèéêíëîïóôõøöù"
            ):
                converted.append(_convert_tcvn3_word(tok))
            else:
                converted.append(tok)
        result = "".join(converted)
        return unicodedata.normalize("NFC", result), encoding

    return unicodedata.normalize("NFC", text), encoding


def fold_vietnamese_diacritics(text: str) -> str:
    """Fold Vietnamese text into ASCII lowercase (removing tone marks and mapping đ/Đ -> d)."""
    if not text:
        return ""
    nfc, _ = normalize_vietnamese_text(text)
    replaced = nfc.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", replaced)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped).lower()
