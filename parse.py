import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GraphRec")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--model_name', type=str, default='CSA4Rec',
                        help="model names")
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--recdim', type=int,default=256,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=0.01,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=1,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=1,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--data_name', type=str,default='Beauty',
                        help="available datasets: [Beauty,ml-100k,yelp2018,Toys_and_Games]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--distill_userK", type=int, default=3, help="distill K")
    parser.add_argument("--distill_uuK", type=int, default=5, help="distill K")
    parser.add_argument("--distill_itemK", type=int, default=3, help="distill K")
    parser.add_argument("--distill_iiK", type=int, default=5, help="distill K")
    parser.add_argument("--distill_layers", type=int, default=5, help="distill number of layers")
    parser.add_argument("--distill_thres", type=float, default=-1, help="distill threshold")
    parser.add_argument("--uuii_thres", type=float, default=-1, help="distill threshold")
    parser.add_argument("--uu_lambda", type=float, default=100, help="lambda for ease")
    parser.add_argument("--ii_lambda", type=float, default=200, help="lambda for ease")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--activation', type=str, default='sigmoid', help='sigmoid')
    parser.add_argument("--resnet", type=float, default=0.001, help="resnet")
    parser.add_argument("--l2", type=bool, default=True, help="l2 norm")
    parser.add_argument("--resnet2", type=float, default=0.01, help="resnet2")
    parser.add_argument("--uu", type=int, default=5, help="uu")
    parser.add_argument("--ii", type=int, default=5, help="ii")




    return parser.parse_args()
