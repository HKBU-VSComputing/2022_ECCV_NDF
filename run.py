from IPython.terminal.embed import embed

from lib.config import args, cfg


def run_dataset():
    import tqdm

    from lib.datasets import make_data_loader

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    import time

    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.networks.renderer import make_renderer
    from lib.utils import net_utils

    cfg.perturb = 0
    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False, current_epoch=100)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        # frame_index = int(batch['meta'][0].split('_')[-3])
        # frame_index = int(batch['meta'][0].split('/')[-1].split('.')[0])
        # if frame_index!=750:
        #     continue
        # if frame_index>=1500:
        #     break
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        batch['step'] = 100000
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch)
    # embed()
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.datasets import make_data_loader
    from lib.networks.renderer import make_renderer
    from lib.utils import net_utils
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer
    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)

    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch)


def run_light_stage():
    from lib.utils.light_stage import ply_to_occupancy
    ply_to_occupancy.ply_to_occupancy()
    # ply_to_occupancy.create_voxel_off()


def run_evaluate_nv():
    from lib.datasets import make_data_loader
    import tqdm

    from lib.evaluators import make_evaluator
    import tqdm

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        evaluator.evaluate(batch)
    evaluator.summarize()


if __name__ == '__main__':
    globals()['run_' + args.type]()
