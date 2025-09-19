import argparse
import sys

def parse_args(args):
        parser = argparse.ArgumentParser(prog='train', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='tub data for training')
        parsed_args = parser.parse_args(args)
        return parsed_args

if __name__ == '__main__':
    print(sys.argv)
    args = parse_args(sys.argv[1:])
    args.tub = ','.join(args.tub)

    from pipeline.training import train
    train(args.tub)

