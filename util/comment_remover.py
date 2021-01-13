""" Strip comments and docstrings from a file.
"""
import re


def remove_python_style_comments(text):
    new_text = text
    regex = re.compile(r"((\/\*)|(\"\"\"))[\s\S]*?((\*\/)|(\"\"\"))|([^:]|^)((\/\/)|#).*?$", re.MULTILINE | re.DOTALL)

    while True:
        old = new_text
        new_text = regex.sub("", new_text)

        if old == new_text:
            break
    return new_text
