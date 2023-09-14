import argparse
import torch


def parameter_parser():
    ap = argparse.ArgumentParser(description="MyModel.")

    ap.add_argument('--dataset', type=str, default='advogato', help='Data set. Default is advogato.')
    ap.add_argument('--node', type=str, default='random', help='Node feature (random | node2vec).')
    ap.add_argument('--node_dim', type=int, default=1024, help='Dimension of init node features. Default is 1024.')
    ap.add_argument('--edge', type=str, default='random', help='Edge feature (random | one-hot).')
    ap.add_argument('--edge_dim', type=int, default=1024, help='Dimension of init edge features. Default is 1024.')
    ap.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden embeddings. Default is 128.')
    ap.add_argument('--num_layers', type=int, default=1, help='Number of the layers. Default is 1.')
    ap.add_argument('--lr', type=float, default=0.005, help='Learning rate. Default is 0.005.')
    ap.add_argument('--weight_decay', type=float, default=0.001, help='Learning rate. Default is 0.0005.')
    ap.add_argument('--dropout', type=float, default=0.5, help='Dropout rate. Default is 0.5.')
    ap.add_argument('--epochs', type=int, default=200, help='Number of epochs. Default is 100.')
    ap.add_argument('--test_size', type=float, default=0.2, help='Test size. Default is 0.2.')
    ap.add_argument('--seed', type=int, default=35213, help='Seed.')
    ap.add_argument('--no_self_loop', action='store_true', default=False,
                    help='Remove self loop or not. Default is False (Not remove).')
    ap.add_argument('--no_cuda', action='store_false', default=True,
                    help='Using CUDA or not. Default is True (Using CUDA).')

    args, _ = ap.parse_known_args()
    args.device = torch.device('cuda:0' if args.no_cuda and torch.cuda.is_available() else 'cpu')
    # args.no_self_loop = True
    return args
