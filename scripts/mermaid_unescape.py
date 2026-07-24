"""MkDocs hook: unescape Mermaid source in rendered HTML.

pymdownx.superfences fence_div_format HTML-escapes the mermaid source
(e.g. `->>` becomes `--&gt;&gt;`), which breaks mermaid.js parsing.
This hook runs after Markdown→HTML conversion and unescapes the content
of every <div class="mermaid"> so mermaid.js receives clean source.
"""


def on_page_content(html, page, config, files):
    import html as html_module
    import re

    def unescape_mermaid(match):
        inner = match.group(1)
        return '<div class="mermaid">' + html_module.unescape(inner) + '</div>'

    return re.sub(
        r'<div class="mermaid">(.*?)</div>',
        unescape_mermaid,
        html,
        flags=re.DOTALL,
    )
