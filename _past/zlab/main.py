from common.args import parse_args
from wrapper import get_sd3_agent, get_dreambooth_dataloader

import pprint

def main(args):
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(vars(args))
    
    # load dataset
    dreambooth_dataloader = get_dreambooth_dataloader(args)
    batch = next(iter(dreambooth_dataloader))
    # print(batch["pixel_values"].shape, batch["prompts"])
    batch["prompts"] = ["a"] * args.train_batch_size
    
    # load model
    sd3_agent = get_sd3_agent(args)
    print(sd3_agent.loss(batch))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
