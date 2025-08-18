def truncate(text: str, max_chars: int) -> str:
    """Corta un texto largo sin partir palabras, añade … si excede."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"

# ---- Markdown helpers ----

def bold(text: str) -> str:
    return f"**{text}**"

def italic(text: str) -> str:
    return f"*{text}*"

def link(title: str, url: str) -> str:
    return f"[{title}]({url})"

def code(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"
