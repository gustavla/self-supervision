

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('-d', '--device', default='/gpu:0', type=str)
    #parser.add_argument('-n', '--number', default=1, type=int)
    parser.add_argument('-o', '--output', default='output', type=str)
    parser.add_argument('-l', '--limit', type=str)
    parser.add_argument('-i', '--iterations', type=int, default=80000)
    args = parser.parse_args()

    task = args.task
    model_fn = args.model

    if task == 'voc2007-classification':
        from .voc_classification import train_and_test
        train_and_test(model_filename=model_fn, output_dir=args.output, device=args.device,
                    time_limit=args.limit, iterations=args.iterations)

if __name__ == '__main__':
    main()
