import re


def split_markdown_at_headings(markdown):
    pattern = r"(#{1,6}\s+.+?(?:\n|$))"
    sections = re.split(pattern, markdown)
    section_text = ""
    section_list = []
    for section_new in sections:
        if not section_new.startswith("#"):
            section_text += section_new
        else:
            section_text = section_text.strip()
            if len(section_text) > 10:
                section_list.append(section_text)
                section_text = section_new
            else:
                section_text += section_new
    section_list.append(section_text)
    return section_list
