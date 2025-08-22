# tests/test_wiki_fetch.py
from agent.tools.wiki import fetch_fullpage

TAXA_OK = [
    "Macaca fuscata",      # Japanese macaque
    "Cebus capucinus",     # White-headed capuchin
    "Aotus nigriceps",     # Black-headed night monkey
    "Cacajao calvus",      # Bald uakari
]

TAXA_NOT_FOUND = [
    "Macaca fuscataxyz",
]

def check_page(page: dict, latin: str):
    print(f"\n=== {latin} ===")
    print("status:", page.get("status"))
    assert page.get("status") in {"ok", "not_found", "error"}

    if page.get("status") != "ok":
        print("error:", page.get("error"))
        return

    title = page["title"]
    url = page["url"]
    text = page["plain_text"]
    infobox = page["infobox"]

    print("title:", title)
    print("url:", url)
    print("plain_len:", len(text))
    print("infobox_keys:", list(infobox.keys())[:8])
    print("preview:", (text[:300] + "…") if len(text) > 300 else text)

    # Aserciones suaves
    assert title and isinstance(title, str)
    assert url.startswith("https://")
    assert isinstance(text, str) and len(text) > 200, "plain_text demasiado corto"
    assert isinstance(infobox, dict)

def main():
    for latin in TAXA_OK:
        page = fetch_fullpage(latin)
        check_page(page, latin)

    for latin in TAXA_NOT_FOUND:
        page = fetch_fullpage(latin)
        print(f"\n=== {latin} (not-found esperado) ===")
        print("status:", page.get("status"))
        assert page.get("status") in {"not_found", "error"}

if __name__ == "__main__":
    main()
