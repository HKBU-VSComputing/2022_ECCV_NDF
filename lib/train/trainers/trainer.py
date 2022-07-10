import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank)
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta' or k == 'split':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        # from IPython import embed
        # embed()

        # time0 = time.time()
        # flag = 200
        # tempBatch = []
        # tempIteration = []
        # for iteration, batch in enumerate(data_loader):
        #     tempIteration.append(iteration)
        #     # tempBatch.append(batch)
        #     flag -= 1
        #     if flag < 1:
        #         break
        # time1 = time.time()
        # print('psbody 0 worker 读取10次数据:', time1 - time0) # 11.5
        # print('psbody 16 worker 读取200次数据:', time1 - time0) # 81.27633929252625
        # print('psbody 16 worker 读取500次数据:', time1 - time0) # 180.27633929252625
        # print('blender 16 worker 读取10次数据:', time1 - time0) # 6.740990400314331
        # print('blender 16 worker 读取200次数据:', time1 - time0) # 37.349268198013306
        # print('blender 16 worker 读取500次数据:', time1 - time0) # 83.72680854797363
        # print('blender 32 worker 读取500次数据:', time1 - time0) # 78.68299198150635
        # print('open3d 2 worker 读取200次数据:', time1 - time0) # 52.868438720703125. 虽然是两个worker，但是htop看cpu也满了
        # print('blender+canonical 8 worker 读取200次数据:', time1 - time0)  # 49.935051679611206


        for iteration, batch in enumerate(data_loader):
            # time0 = time.time()
            # for iteration, batch in zip(tempIteration, tempBatch):
            batch = self.to_cuda(batch)

            data_time = time.time() - end
            iteration = iteration + 1

            try:
                batch['step'] = recorder.step
            except:
                pass

            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter -
                                                                  1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter -
                                                                iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(
                    ['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string,
                                                       str(recorder), lr,
                                                       memory)
                print(training_state)

            if iteration % cfg.record_interval == 0 or iteration == (max_iter -
                                                                     1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
