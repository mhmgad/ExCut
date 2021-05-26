import codecs, argparse


def convert_ere2eer(input_filename, output_filename):
    """
    Converts file from e1 r e2 to e1 e2 r, this is required for
    :param input_filename: in format e1 r e2
    :param output_filename: e1 e2 r
    :return:
    """
    with codecs.open(input_filename, "r") as input_file:
        with codecs.open(output_filename, "w") as output_file:
            for line in input_file:
                line = line.strip().split('\t')
                if len(line)<3:
                    output_file.write('\t'.join([str(c) for c in line])+'\n')
                    continue

                line = [line[0],line[2],line[1]]
                # print(line)
                output_file.write('\t'.join([str(c) for c in line])+'\n')







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="input kg file")
    parser.add_argument("-o","--output", help="output file")
    parser.add_argument("-er", "--ere2eer", help="converts from entity relation entity to entity entity relation", action="store_true")
    args = parser.parse_args()

    if args.ere2eer:
        convert_ere2eer(args.input, args.output)
