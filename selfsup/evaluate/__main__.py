

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('-d', '--device', default='/gpu:0', type=str)
    #parser.add_argument('-n', '--number', default=1, type=int)
    parser.add_argument('-o', '--output', default='output', type=str)
    parser.add_argument('-l', '--limit', type=str)
    args = parser.parse_args()

    task = args.task
    model_fn = args.model

    if task == 'voc2007-classification':
        from .voc_classification import train, test
        args = dict(model_filename=model_fn, output_dir=args.output, device=args.device,
                    time_limit=args.limit)
        train(**args)
        test(**args)

if __name__ == '__main__':
    main()
