"""
This modules contains some frequently required output functions
"""

import codecs, os
from excut.utils.logging import logger
from urllib.parse import unquote


def add_arrows(s):
    if not s.startswith('<'):
        s='<'+s
    if not s.endswith('>'):
        s+='>'
    return s

def write_triples(triples_iteratable, output_filename, surround=False, unquote_urls=True):
    """
    Exports triples from a source to file as tsv.

    :param unquote_urls:
    :param triples_iteratable:
    :param output_filename:
    :param surround:
    :return:
    """
    logger.info("Writing triples to %s" % output_filename)
    with codecs.open(output_filename, "w") as output_file:
        for t in triples_iteratable:

            if unquote_urls:
                t=map(unquote,t)
            if surround:
                t=map(add_arrows,t)
            output_file.write('\t'.join(t))
            output_file.write('\n')
    logger.info("Done writing triples to %s" % output_filename)


def write_triples_as_predicates(triples_iteratable, output_filename, surround=False):
    logger.info("Writing triples to %s" % output_filename)
    with codecs.open(output_filename, "w") as output_file:
        for t in triples_iteratable:
            if surround:
                t=map(add_arrows,t)
            output_file.write('%s(%s,%s)'%(t[1],t[0],t[2]))
            output_file.write('\n')
    logger.info("Done writing triples to %s" % output_filename)

def write_vectors(vectors, output_filename):
    logger.info("Writing vectors to %s" % output_filename)
    with codecs.open(output_filename, "w") as output_file:
        for t in vectors:
            output_file.write('\t'.join(map(str,t)))
            output_file.write('\n')
    logger.info("Done writing vectors to %s" % output_filename)


def dump_dict(dict_to_dump,dict_filename, overwrite):
    if overwrite or not os.path.exists(dict_filename):
        logger.info("Dump to %s" % dict_filename)
        with codecs.open(dict_filename, "w") as output_file:
            output_file.write(str(len(dict_to_dump.keys())) + '\n')
            for k, v in dict_to_dump.items():
                output_file.write(str(k) + '\t' + str(v) + '\n')
        return True
    return False



