import re


NEWLINE_PTR = re.compile(r'(?:"[^"]*"|.)+')  # \n outside of quotes (https://stackoverflow.com/questions/24018577/parsing-a-string-in-python-how-to-split-newlines-while-ignoring-newline-inside)
COMMENT_PTR = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|#[^"\']*?(?=\n)|#.*?(".*?"|\'.*?\').*?(?=\n)', re.DOTALL|re.MULTILINE) 


def split_newlines(s):
    """
    Split based on new lines (\n) outside of quotes.

    Note that this coalesces several newlines into one,
    as blank lines are ignored. To avoid that, give a null case:

    (?:"[^"]*"|.)+|(?!\Z)
    """
    return re.findall(NEWLINE_PTR, s)

def remove_comments(source):
    return re.sub(r"\n\n+", "\n\n", re.sub(COMMENT_PTR, "", source))