from configs import *
import numpy as np
import torch
from ignite.metrics import Metric
import time
import logging

import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def euclidean_dist_cpu(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = np.power(x, 2).sum(1)
    xx = np.reshape(xx, [xx.shape[0], 1])
    xx = xx.repeat(n, axis=1)
    yy = np.power(y, 2).sum(1)
    yy = np.reshape(yy, [yy.shape[0], 1])
    yy = yy.repeat(m, axis=1).T
    dist = xx + yy
    dist -= 2 * np.dot(x, y.T)
    dist = np.sqrt(np.clip(dist, 1e-12, dist.max()))
    return dist

def cos_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    xx = x/x.norm(dim=1)[:,None]
    yy = y/y.norm(dim=1)[:,None]
    dist = torch.mm(xx,yy.t())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return 1-dist

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, re_rank = False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.re_rank = re_rank
        self.count = 0



    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        # f = open(self.pkl_path, "rb")
        # feats = pickle.load(f)


        feats = torch.cat(self.feats, dim=0)
        fnorm = torch.norm(feats,p=2,dim=1,keepdim=True)
        feats = feats.div(fnorm.expand_as(feats))
        # # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())

        qf = qf.cpu().numpy()
        gf = gf.cpu().numpy()



        distmat_q_g = euclidean_dist_cpu(qf,gf)


        # raw_data = {"qf":qf,
        #          "gf":gf,
        #          #"distmat_q_g":distmat_q_g,
        #          "q_pids":q_pids,
        #          "g_pids":g_pids,
        #          "q_camids":q_camids,
        #          "g_camids":g_camids
        #         }
        #         #save distmat
        # f = open('/home/gtp_cgy/ivg/dataset/LRR/msmt_train.pkl','wb+')
        # pickle.dump(raw_data,f)
        # f.close()

        pids = np.asarray(self.pids)
        camids = np.asarray(self.camids)

        #print(len(pids))


        # raw_data = {
        #     "feats": feats,
        #     "pids": pids,
        #     "camids": camids
        # }
        #
        #


        # exit()


        start = time.time()
        if self.re_rank:
        #distmat_cos = cos_dist(qf,gf)
            distmat_q_q = euclidean_dist_cpu(qf,qf)
            distmat_g_g = euclidean_dist_cpu(gf,gf)
            # distmat_q_q = distmat_q_q.cpu().numpy()
            # distmat_g_g = distmat_g_g.cpu().numpy()
            distmat = re_ranking(distmat_q_g,distmat_q_q,distmat_g_g )
            duration = time.time()-start
            print(f"Re-ranking runing in {duration}")
        else:
            distmat = distmat_q_g
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
    
def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)
        #pdb.set_trace()

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch

        img = img.cuda()
        target = target.cuda()
        score, feat = model(img)

        loss = loss_fn(score, feat, target)

        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def fliplr(img):
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
#             feat = model(data)
            # ######### fliplr ####
            data_f = fliplr(data)
            feat = model(data)
            feat_f = model(data_f)
            feat = feat + feat_f



            return feat, pids, camids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query
):
    log_period = SOLVER_LOG_PERIOD
    checkpoint_period = SOLVER_CHECKPOINT_PERIOD
    eval_period = SOLVER_EVAL_PERIOD
    output_dir = OUTPUT_DIR
    device = MODEL_DEVICE
    epochs = SOLVER_MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, MODEL_NAME,  n_saved=5, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] \nLoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'],
                                engine.state.metrics['avg_acc'], scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if ((engine.state.epoch % eval_period == 0) or (engine.state.epoch == epochs)) and (engine.state.epoch > 0.5*epochs):
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))

            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    trainer.run(train_loader, max_epochs=epochs)