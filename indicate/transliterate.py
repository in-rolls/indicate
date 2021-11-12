import sys
import argparse

from .hindi2english import HindiToEnglish

hindi2english = HindiToEnglish.transliterate


def main(argv=sys.argv[1:]):
    title = "Transliterate from Hindi to English"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--type", default=None, help="hin2eng")
    parser.add_argument("--input", default=None, help="Hindi text")
    args = parser.parse_args(argv)
    print(args)
    if not args.input or not args.type:
        return -1

    output = ""
    if args.type == "hin2eng":
        output = hindi2english(args.input)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
