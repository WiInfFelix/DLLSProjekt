from histogramm import Histogrammer
from Regexer import TextTagger
from pprint import pprint
from SpanPrinter import SpanPrint


def main():
    hst = Histogrammer()

    tt = TextTagger()

    sp = SpanPrint()

    sp.print_spans()

    # spans = sp.get_article_spans()


if __name__ == '__main__':
    main()
