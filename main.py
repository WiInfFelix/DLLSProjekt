from histogramm import Histogrammer
from Regexer import TextTagger
from pprint import pprint
from SpanPrinter import SpanPrint


def main():
    hst = Histogrammer()

    tt = TextTagger()

    # pprint(tt.tuple_articles_and_indices())

    # pprint(tt.tuple_words_indices())

    sp = SpanPrint()

    sp.print_spans()


if __name__ == '__main__':
    main()
