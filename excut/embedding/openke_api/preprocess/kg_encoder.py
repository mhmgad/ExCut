import argparse
import codecs
import os
import threading

from openke_api.preprocess.format_utils import convert_ere2eer
from excut.utils.logging import logger
from excut.kg.kg_triples_source import FileTriplesSource
from excut.utils.output_utils import dump_dict


class Counter:
    def __init__(self, start_value=0):
        self.i = start_value
        # create a lock
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        # acquire/release the lock when updating self.i
        with self.lock:
            result = self.i
            self.i += 1
            return result


class Encoder():

    def __init__(self, kg_encoding_folder, immutable=False):

        self.immutable = immutable
        entities_dict_filename = os.path.join(kg_encoding_folder, "entity2id.txt")
        relations_dict_filename = os.path.join(kg_encoding_folder, "relation2id.txt")

        # self.drop_prefix=drop_prefix

        self.entities2ids = self.load_dict(entities_dict_filename)
        self.ids2entities = self.load_dict(entities_dict_filename, flipped=True)

        self.relations2ids = self.load_dict(relations_dict_filename)
        self.ids2relations = self.load_dict(relations_dict_filename, flipped=True)

        self.entity_counter = Counter(max(self.ids2entities.keys()) if self.ids2entities.keys() else 0)
        self.relation_counter = Counter(max(self.ids2relations.keys()) if self.ids2relations.keys() else 0)

    def _entity2id(self, entity):
        # TODO: Not thread-safe
        if not entity in self.entities2ids:
            if self.immutable:
                raise
            id = self.entity_counter.next()
            self.entities2ids[entity] = id
            self.ids2entities[id] = entity

        return self.entities2ids[entity]

    def conv_entities2ids(self, entities):
        return [self._entity2id(ent) for ent in entities]

    def _relation2id(self, relation):
        # TODO: Not thread-safe
        if not relation in self.relations2ids:
            id = self.relation_counter.next()
            self.relations2ids[relation] = id
            self.ids2relations[id] = relation

        return self.relations2ids[relation]

    def conv_relations2ids(self, relations):
        return [self._entity2id(ent) for ent in relations]

    def strings2ids_from_file(self, input_filename, output_filename=None, column_offset=0):
        logger.info("encode triples from %s \n\t save to %s" % (input_filename, output_filename))
        self.strings2ids(FileTriplesSource(input_filename, column_offset), output_filename)

    def strings2ids(self, triples_source, output_filename=None):
        logger.info("Encode triples and save to %s" % (output_filename))
        with codecs.open(output_filename, "w") as output_file:
            # triples_as_list = list(triples_source.triples())
            # output_file.write(str(len(triples_as_list)) + '\n')
            # triples_as_list = list(triples_source.triples())
            output_file.write(str(triples_source.size()) + '\n')
            logger.info("Encoding %i triples!" % triples_source.size())
            for t in triples_source:
                encoded_t = (self._entity2id(t[0]), self._relation2id(t[1]), self._entity2id(t[2]))
                output_file.write('\t'.join([str(c) for c in encoded_t]) + '\n')
        logger.info("Done encoding triples!")
        return output_filename

    def load_dict(self, filename, flipped=False):
        if os.path.exists(filename):
            # logger.info("Exists!"+ str(os.path.exists(filename)))
            logger.debug("Loading %s into dictionary! (Flipped: %r)" % (filename, flipped))
            with open(filename) as fh:
                # fh.readline()
                lines = (line.split(None, 1) if '\t' in line else ['q', -1] for line in fh)
                if flipped:
                    mapping = dict((int(number), word) for word, number in lines)
                else:
                    mapping = dict((word, int(number)) for word, number in lines)
            logger.debug("Dictionary Size: %i" % len(mapping))
            # print(mapping[68])
            return mapping

        return dict()

    def dump_dicts(self, out_dir, overwrite=False):
        # logger.info("Dump encoding to %s"%out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, True)

        entities_dict_filename = os.path.join(out_dir, "entity2id.txt")
        dump_dict(self.entities2ids, entities_dict_filename, overwrite)

        relations_dict_filename = os.path.join(out_dir, "relation2id.txt")
        dump_dict(self.relations2ids, relations_dict_filename, overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input kg file")
    parser.add_argument("-o", "--output", help="output_folder", default=None)
    parser.add_argument("-f", "--offset", help="column offset", default=0, type=int)
    # parser.add_argument("-en","--encoding_old", help="Some old encoding folder", default=None)
    args = parser.parse_args()

    out_dir = args.output
    column_offset = args.offset

    encoder = Encoder()
    encoder.strings2ids_from_file(args.input, out_dir + "/triple2id.txt",
                                  column_offset=column_offset)
    encoder.dump_dicts(out_dir + "/entity2id.txt", out_dir + "/relation2id.txt")
    convert_ere2eer(out_dir + "/triple2id.txt", out_dir + "/train2id.txt")

    # out_dir="/GW/D5data-11/gadelrab/yago2018/encoded_kg_types"
    #
    # encoder=Encoder(entities_dict_filename='/GW/D5data-11/gadelrab/yago2018/encoded_kg/entity2id.txt',relations_dict_filename='/GW/D5data-11/gadelrab/yago2018/encoded_kg/relation2id.txt')
    # encoder.strings2ids("/GW/D5data-9/gadelrab/yago2018/yagoFacts_types_20k.tsv", out_dir+"/triple2id.txt", column_offset=0)
    # encoder.dump_dicts(out_dir+"/entity2id.txt", out_dir+"/relation2id.txt")
    # convert_ere2eer(out_dir+"/triple2id.txt", out_dir+"/train2id.txt")
