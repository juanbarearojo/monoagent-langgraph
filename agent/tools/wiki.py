# agent/tools/wiki.py
from __future__ import annotations
from typing import Dict, List, Union
import re
import requests
from bs4 import BeautifulSoup, Tag, NavigableString

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE_TMPL = "https://en.wikipedia.org/wiki/{title}"
UA = "MonoAgent/0.1 (contact: you@example.com)"

# Regex para limpiar refs y espacios
_REF_PAT = re.compile(r"\[citation needed\]|\[[0-9]+\]|\[[a-z]\]", re.IGNORECASE)
_WS_PAT = re.compile(r"[ \t\u00A0]+")

def _clean_text(s: str) -> str:
    s = _REF_PAT.sub("", s)           # quita referencias [1], [a], [citation needed]
    s = _WS_PAT.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)  # colapsa saltos de línea múltiples
    return s.strip()

def _html_to_text(el: Union[Tag, NavigableString]) -> str:
    if isinstance(el, NavigableString):
        return _clean_text(str(el))
    for sup in el.select("sup.reference, sup[role='note']"):
        sup.decompose()
    for rm in el.select("table.navbox, table.metadata, figure, style, script"):
        rm.decompose()
    return _clean_text(el.get_text(separator="\n", strip=True))

def _parse_infobox(root: Tag) -> Dict[str, str]:
    def has_infobox_class(c):
        if not c:
            return False
        if isinstance(c, list):
            return any("infobox" in cls for cls in c)
        return "infobox" in c

    table = root.find("table", class_=has_infobox_class)
    if not table:
        return {}
    info: Dict[str, str] = {}
    for tr in table.find_all("tr", recursive=False):
        th, td = tr.find("th"), tr.find("td")
        if not th or not td:
            continue
        key = _clean_text(th.get_text(" ", strip=True)).rstrip(":")
        val = _html_to_text(td)
        if key and val:
            info[key] = val
    return info

def _extract_full_plaintext(html: str) -> str:
    """
    Extrae TODO el texto útil de la página:
    - Párrafos y listas
    - Sin encabezados ni referencias
    - Sin tablas salvo infobox (que se parsea aparte)
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup

    parts: List[str] = []
    for el in content.children:
        if not isinstance(el, Tag):
            continue

        if el.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            continue
        if el.get("id") == "toc" or "toc" in el.get("class", []):
            continue
        if el.name == "table":
            continue
        if el.name in {"figure", "style", "script"}:
            continue

        if el.name == "p":
            txt = _html_to_text(el)
            if txt:
                parts.append(txt)
        elif el.name in {"ul", "ol"}:
            items = [_html_to_text(li) for li in el.find_all("li", recursive=False)]
            items = [i for i in items if i]
            if items:
                parts.append("\n".join(items))
        elif el.name == "div":
            for p in el.find_all("p", recursive=False):
                t = _html_to_text(p)
                if t:
                    parts.append(t)

    return _clean_text("\n\n".join(parts))

def fetch_fullpage(latin: str) -> Dict[str, object]:
    """
    Devuelve:
      {
        "title": str,
        "url": str,
        "plain_text": str,   # TODO el texto útil de la página
        "infobox": Dict[str, str],
        "status": "ok"|"not_found"|"error",
        "error": Optional[str]
      }
    """
    if not latin:
        return {"status": "not_found", "error": "empty_title"}

    title = latin.strip().replace(" ", "_")
    url = WIKI_PAGE_TMPL.format(title=title)

    try:
        r = requests.get(
            WIKI_API,
            headers={"User-Agent": UA},
            params={
                "action": "parse",
                "page": title,
                "prop": "text|displaytitle",
                "format": "json",
                "redirects": 1,
            },
            timeout=12,
        )
        data = r.json()
        if "error" in data or "parse" not in data:
            return {"status": "not_found"}

        parsed = data["parse"]
        display_title_html = parsed.get("displaytitle") or parsed.get("title") or latin
        display_title = _clean_text(BeautifulSoup(display_title_html, "html.parser").get_text(" ", strip=True))
        html = parsed["text"]["*"]

        soup = BeautifulSoup(html, "html.parser")
        content = soup.select_one("div.mw-parser-output") or soup
        infobox = _parse_infobox(content)
        plain_text = _extract_full_plaintext(html)

        return {
            "title": display_title,
            "url": url,
            "plain_text": plain_text,
            "infobox": infobox,
            "status": "ok",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
