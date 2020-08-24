"""
This module offers some data cleaning utils to fix URI represenation
"""

import re
from urllib.parse import urljoin, quote, urlsplit, urlunsplit, unquote
import numpy as np

# regex = re.compile(
#         r'^(?:http|ftp)s?://' # http:// or https://
#         r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
#         r'localhost|' #localhost...
#         r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
#         r'(?::\d+)?' # optional port
#         r'(?:/?|[/?]\S+)$', re.IGNORECASE)

regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?))' #domain...
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def is_valid_url(x):
    return (x.startswith("http:") or x.startswith("https:"))
    # return (re.match(regex, x) is not None)


def in_braces(x):
    return x.startswith('<') and x.endswith('>')


def strip_braces(x):
    if in_braces(x):
        return x[1:-1]
    else:
        return x


def entity_full_url(x, prefix='', quote_it=False):
    x = strip_braces(x)
    if prefix or  quote_it:
        if is_valid_url(x):
            split = list(urlsplit(x))
            split[2] = quote(split[2], encoding='utf-8') if quote_it else split[2]
            return urlunsplit(split)
        else:
            # print("not")
            return urljoin(prefix, quote(x, encoding='utf-8') if quote_it else x)
    else:
        return x


def relation_full_url(x,prefix=''):
    # Nothing done for now, just join prefix

    x = strip_braces(x)
    if is_valid_url(x):
        return x
    return urljoin(prefix,x)

def format_triple(t, prefix='', quote_it=False):
    if len(t)<3:
        return np.array(list(map(lambda e: entity_full_url(e, prefix, quote_it),t)), dtype=object)
    else:
    # t=t.tolist()
        return np.array([entity_full_url(t[0], prefix, quote_it),
        relation_full_url(t[1], prefix),
        entity_full_url(t[2], prefix, quote_it)], dtype=object)




def is_var(x):
    return x.startswith('?')

def is_url(x):
    return x.startswith('http:') or x.startswith("https:")

def _sparql_repr(x):
    if is_var(x):
        return x
    else:
        if in_braces(x):
            return x
        else:
            return '<%s>' % x

def _no_url(s):
    if is_url(s):
        s=urlsplit(s)[2]
        if s.startswith('/'):
            return s[1:]
    return s

def str_tuple(t):
    return '(%s,%s,%s)' %t


def str_tuple_readable(t):
    # t=(_no_url(t[0]),_no_url(t[1]) ,_no_url(t[2]))
    return '(%s,%s,%s)' %t

def _n3_repr(x):
    # return x if is_var(x) else '<%s>' % x
    # TODO drop this method once
    return _sparql_repr(x)


def n3_repr(t):
    return '\t'.join(map(_n3_repr, t))

def sparql_repr(t):
    return '%s.' %' '.join(map(_n3_repr, t))

def valid_id(x):
    if '\\' in x:
        return False
    return True

def valid_id_triple(t):
    return all(valid_id(a) for a in t)

if __name__ == '__main__':

    print(valid_id('Phil_\\u0022Philthy_Animal\\u0022_Taylor'))

    print(relation_full_url('rdf:type', "http://example.org/"))
    print(entity_full_url('<ab a>', "http://example.org/", True))
    print(entity_full_url('http://example.org/a:b/a', True))
    print(entity_full_url('http://example.c.it1/a', True))

    print(entity_full_url('http://example.org/ab%20a','http://example.org',False))
    print(entity_full_url('http://example.org/a%3Ab/a','',False))

    print(entity_full_url('http://example.org/ab%20a', 'http://yago.irg', True))











