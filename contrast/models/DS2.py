import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size


DATA_INCOMPATIBLE_ERR = (
    "dimension of data is incompatible with augmentation method or loss function"
)


class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
            for global branch, N is batch size, for dense branch, N is bs*S^2

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)  # shape: (N, 1+K)
        logits /= self.temperature
        labels = (
            torch.zeros((N,), dtype=torch.long).cuda()
            if torch.cuda.is_available()
            else torch.zeros((N,), dtype=torch.long)
        )  # shape: (N), represents index of pos (i.e., index 0) for each item in N
        return self.criterion(logits, labels)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True
    )


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


class FC_MLP(nn.Module):
    def __init__(self, head_dims=[512, 512, 512, 512, 512, 512, 512, 512, 128]):
        super().__init__()
        layers = []
        prev_dim = head_dims[0]

        for current_dim in head_dims[1 : len(head_dims) - 1]:
            layers.append(
                nn.Linear(in_features=prev_dim, out_features=current_dim, bias=False)
            )
            layers.append(nn.BatchNorm2d(current_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = current_dim

        layers.append(
            nn.Linear(in_features=prev_dim, out_features=head_dims[-1], bias=False)
        )

        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.projector(x)


class RotPredHead(nn.Module):
    def __init__(
        self,
        in_dim=50176,
        inner_1_dim=1024,
        inner_2_dim=512,
        out_dim=512,
        out_class_num=4,
    ):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=in_dim, out_features=inner_1_dim, bias=False
        )
        self.bn1 = nn.BatchNorm2d(inner_1_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(
            in_features=inner_1_dim, out_features=inner_2_dim, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inner_2_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(
            in_features=inner_2_dim, out_features=out_dim, bias=True
        )
        self.linear4 = nn.Linear(
            in_features=out_dim, out_features=out_class_num, bias=True
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # following the baseline, linear 3 and linear 4 are not equipped with normalization and activation layers
        x = self.linear3(x)
        x = self.linear4(x)
        return x


def ds2_loss(q, k, q_bb, k_bb, coord_q, coord_k):
    """
    coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    # coord_* values are in the range of [0.0, 1.0]
    # If input view is resized to 224, then  N: bs, C: 256, C_BB: 512, H: 7, W: 7
    N, C, H, W = q.shape
    _, C_BB, _, _ = q_bb.shape

    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)  # combine H and W, new dimensions: (N, C, H*W)
    k = k.view(N, C, -1)
    q_bb = q_bb.view(N, C_BB, -1)
    k_bb = k_bb.view(N, C_BB, -1)

    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)

    # [bs, 1, 1]
    q_bin_diag = torch.sqrt(q_bin_width**2 + q_bin_height**2)
    k_bin_diag = torch.sqrt(k_bin_width**2 + k_bin_height**2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = (
        torch.arange(0.0, float(W), dtype=coord_q.dtype, device=coord_q.device)
        .view(1, 1, -1)
        .repeat(1, H, 1)
    )
    y_array = (
        torch.arange(0.0, float(H), dtype=coord_q.dtype, device=coord_q.device)
        .view(1, -1, 1)
        .repeat(1, 1, W)
    )

    # [bs, 7, 7]
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

    # [bs, 49, 49]
    dist_center = (
        torch.sqrt(
            (center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
            + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2
        )
        / max_bin_diag
    )
    pos_mask = (
        (dist_center < 0.1).float().detach()
    )  # negative pairs does not contribute to loss

    # [bs, 49, 49], each element of 49*49 is a cosine score
    logit = torch.bmm(q.transpose(1, 2), k)

    # add top-k similar pairs from logit_bb to pos_mask
    logit_bb = torch.div(torch.bmm(q_bb.transpose(1, 2), k_bb), C_BB)
    top_k = 1
    max_k_indices = (
        torch.topk(logit_bb.view(N, -1), top_k).indices.detach().tolist()
    )  # shape: array of size: (bs, k)
    # random.shuffle(max_k_indices)
    for batch_index, locs in enumerate(max_k_indices):
        for item in locs:
            x, y = (
                int(item / logit.shape[-1]),
                int(item % logit.shape[-1]),
            )
            pos_mask[batch_index, x, y] = 1.0

    # calculate loss
    loss = (logit * pos_mask).sum(-1).sum(-1) / (
        pos_mask.sum(-1).sum(-1) + 1e-6
    )  # shape: bs

    return -2 * loss.mean()  # shape: scalar


def distaug_mocov2_loss(
    q,
    k,
    ins_queue,
    class_aware,
    ins_queue_target=None,
    target=None,
):
    # q,k: shape (bs, c)

    # positive pair similarities
    pos = torch.einsum("nc, nc->n", [q, k]).unsqueeze(1)  # shape: (bs, 1)

    # negative pairs
    neg = torch.einsum(
        "nc,ck->nk",
        [
            q,
            ins_queue.clone().detach(),
        ],
    )  # shape: (bs, K), where K is queue size

    return (
        ContrastiveHead(temperature=0.2)(
            pos,
            neg
            * torch.cat(
                [
                    torch.where(
                        ins_queue_target == v,
                        torch.zeros_like(ins_queue_target),
                        torch.ones_like(ins_queue_target),
                    ).unsqueeze(0)
                    for v in target.detach().tolist()
                ],
                dim=0,
            ).detach()
            + torch.cat(
                [
                    torch.where(ins_queue_target == v, -1e10, 0.0).unsqueeze(0)
                    for v in target.detach().tolist()
                ],
                dim=0,
            ),
        )
        if class_aware
        else ContrastiveHead(temperature=0.2)(pos, neg)
    )


def rotpred_loss(pred, actual_class):
    # pred.shape: (bs, #classes)
    temperature = 0.2

    pred /= temperature
    labels = (
        actual_class.to(torch.long).cuda()
        if torch.cuda.is_available()
        else actual_class.to(torch.long)
    )

    return nn.CrossEntropyLoss()(pred, labels)


def simclr_loss(feat1, feat2, temp=0.2, target=None, class_aware=False):
    N, C = feat1.shape  # shape: (bs,256)

    out = torch.cat([feat1, feat2], dim=0)  # shape: (2*bs, 256)
    sim_matrix = torch.exp(
        torch.mm(out, out.t().contiguous()) / temp
    )  # shape: (2*bs, 2*bs)

    if class_aware:
        # remove same class views from negative samples
        _, inverse_indices, counts = torch.unique(
            target, return_inverse=True, return_counts=True
        )

        all_combs = None
        dupValuePositions = torch.where(counts > 1)[0]
        for position in dupValuePositions.cpu().numpy():
            dupValueIndices = torch.where(inverse_indices == position)[0]
            new_combs = torch.combinations(dupValueIndices, r=2).cpu().numpy()
            all_combs = (
                new_combs
                if all_combs is None
                else np.concatenate((all_combs, new_combs), axis=0)
            )

        same_class_mask = torch.ones(
            target.shape[0], target.shape[0]
        )  # views from same class will be set to value 0, except with itself

        if all_combs is not None:
            for pair in all_combs:
                [i, j] = pair
                same_class_mask[i, j] = 0
                same_class_mask[j, i] = 0
            same_class_mask = torch.cat((same_class_mask, same_class_mask), dim=1)
            same_class_mask = torch.cat((same_class_mask, same_class_mask), dim=0).to(
                sim_matrix.device
            )  # shape: (2*bs, 2*bs)

            sim_matrix = (
                sim_matrix * same_class_mask
            )  # remove contribution from views of same image class

    mask = (
        torch.ones_like(sim_matrix) - torch.eye(2 * N, device=sim_matrix.device)
    ).bool()  # shape: (2*bs, 2*bs)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * N, -1)  # shape: (2*bs, 2*bs-1)

    pos_sim = torch.exp(torch.sum(feat1 * feat2, dim=-1) / temp)  # shape: bs
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # shape: 2*bs
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def cosine_loss(x, y):
    return -2.0 * torch.einsum("nc, nc->n", [x, y]).mean()


def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


class DS2(nn.Module):
    def __init__(self, base_encoder, args):
        super().__init__()

        # parse arguments
        self.pixpro_p = args.pixpro_p
        self.pixpro_momentum = args.pixpro_momentum
        self.pixpro_clamp_value = args.pixpro_clamp_value
        self.pixpro_transform_layer = args.pixpro_transform_layer

        self.dense_loss_weight = args.dense_loss_weight
        self.dense_loss_func = args.dense_loss_func
        self.instance_loss_weight = args.instance_loss_weight
        self.instance_loss_func = args.instance_loss_func
        self.instance_branch_class_aware = args.instance_branch_class_aware
        self.head_dims = args.head_dims

        #######
        ###  create online & momentum encoder
        #######
        self.encoder = base_encoder(
            low_dim=args.feature_dim,
            in_channel=args.in_channel,
        )

        if not (
            self.dense_loss_weight == 0.0
            and self.instance_loss_weight > 0.0
            and self.instance_loss_func in ["DistAug_SimCLR", "SimCLR", "RotPred"]
        ):
            self.encoder_k = base_encoder(
                low_dim=args.feature_dim,
                in_channel=args.in_channel,
            )  # create the encoder_k (momentum)

            for param_q, param_k in zip(
                self.encoder.parameters(), self.encoder_k.parameters()
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        if torch.cuda.is_available():
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            if not (
                self.dense_loss_weight == 0.0
                and self.instance_loss_weight > 0.0
                and self.instance_loss_func in ["DistAug_SimCLR", "SimCLR", "RotPred"]
            ):
                nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        #######
        ###  create dense-branch online & momentum projector
        #######
        if self.dense_loss_weight > 0.0:
            self.projector_c4 = (
                Proj_Head(in_dim=256) if args.arch == "resnet18" else Proj_Head()
            )
            self.projector_c5 = (
                Proj_Head(in_dim=512) if args.arch == "resnet18" else Proj_Head()
            )

            self.projector_k_c4 = (
                Proj_Head(in_dim=256) if args.arch == "resnet18" else Proj_Head()
            )  # create the encoder_k (momentum)
            self.projector_k_c5 = (
                Proj_Head(in_dim=512) if args.arch == "resnet18" else Proj_Head()
            )  # create the encoder_k (momentum)

            for param_q, param_k in zip(
                self.projector_c4.parameters(), self.projector_k_c4.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            for param_q, param_k in zip(
                self.projector_c5.parameters(), self.projector_k_c5.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            if torch.cuda.is_available():
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_c4)
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k_c4)
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_c5)
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k_c5)

        self.K = int(
            args.num_instances * 1.0 / get_world_size() / args.batch_size * args.epochs
        )
        self.k = int(
            args.num_instances
            * 1.0
            / get_world_size()
            / args.batch_size
            * (args.start_epoch - 1)
        )

        #######
        # if use instance branch
        # depending on architecture choice, create (1) online & momentum projector_instance; (2) predictor
        #######
        if self.instance_loss_weight > 0.0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            if self.instance_loss_func == "RotPred":
                self.rot_MLP2d = MLP2d(
                    in_dim=512, inner_dim=256, out_dim=64
                )  # for shrinking channel-wise dimension
                self.rot_predictor = RotPredHead(in_dim=6272)  # 64 * 49 * 2
                if torch.cuda.is_available():
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.rot_MLP2d)
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.rot_predictor)
            elif self.instance_loss_func == "DistAug_SimCLR":
                head_dims = [int(dim) for dim in self.head_dims.split(",")]
                self.fc_mlp = FC_MLP(head_dims=head_dims)
                if torch.cuda.is_available():
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.fc_mlp)
            elif self.instance_loss_func == "SimCLR":
                self.projector_instance = (
                    Proj_Head(in_dim=512) if args.arch == "resnet18" else Proj_Head()
                )
                if torch.cuda.is_available():
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
            else:
                # instance_loss_func in ["Cosine", "MOCOv2", "DistAug_MOCOv2"]
                self.projector_instance = (
                    Proj_Head(in_dim=512) if args.arch == "resnet18" else Proj_Head()
                )
                self.projector_instance_k = (
                    Proj_Head(in_dim=512) if args.arch == "resnet18" else Proj_Head()
                )

                if self.instance_loss_func == "Cosine":
                    # Cosine loss means BYOL-alike loss
                    self.predictor = Pred_Head()
                elif self.instance_loss_func in ["DistAug_MOCOv2", "MOCOv2"]:
                    # queue
                    self.ins_queue_len = 4096  # multiple of bs = 2*128 = 256
                    feat_dim = 256

                    self.register_buffer(
                        "ins_queue", torch.randn(feat_dim, self.ins_queue_len)
                    )
                    self.ins_queue = nn.functional.normalize(self.ins_queue, dim=0)
                    self.register_buffer(
                        "ins_queue_ptr", torch.zeros(1, dtype=torch.long)
                    )

                    if self.instance_branch_class_aware:
                        self.register_buffer(
                            "ins_queue_target",
                            torch.randint(100, 200, (self.ins_queue_len,)),
                        )  # randomly initialize meaningless image classes (100 is higher than total class numbers)
                        self.register_buffer(
                            "ins_queue_target_ptr", torch.zeros(1, dtype=torch.long)
                        )

                for param_q, param_k in zip(
                    self.projector_instance.parameters(),
                    self.projector_instance_k.parameters(),
                ):
                    param_k.data.copy_(param_q.data)
                    param_k.requires_grad = False

                if torch.cuda.is_available():
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance_k)
                    if self.instance_loss_func == "Cosine":
                        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        #######
        ###  create value_transform module
        #######
        if self.dense_loss_weight > 0.0 and self.dense_loss_func in ["DS2"]:
            if self.pixpro_transform_layer == 0:
                self.value_transform = Identity()
            elif self.pixpro_transform_layer == 1:
                self.value_transform = conv1x1(in_planes=256, out_planes=256)
            elif self.pixpro_transform_layer == 2:
                self.value_transform = MLP2d(in_dim=256, inner_dim=256, out_dim=256)
            else:
                raise NotImplementedError

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = (
            1.0
            - (1.0 - self.pixpro_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.0
        )
        self.k = self.k + 1

        #######
        ###  update momentum encoder
        #######
        if not (
            self.dense_loss_weight == 0.0
            and self.instance_loss_weight > 0.0
            and self.instance_loss_func in ["DistAug_SimCLR", "SimCLR", "RotPred"]
        ):
            for param_q, param_k in zip(
                self.encoder.parameters(), self.encoder_k.parameters()
            ):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                    1.0 - _contrast_momentum
                )

        #######
        ###  update dense-branch momentum projector
        #######
        if self.dense_loss_weight > 0.0:
            for param_q, param_k in zip(
                self.projector_c4.parameters(), self.projector_k_c4.parameters()
            ):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                    1.0 - _contrast_momentum
                )
            for param_q, param_k in zip(
                self.projector_c5.parameters(), self.projector_k_c5.parameters()
            ):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                    1.0 - _contrast_momentum
                )

        #######
        ###  update instance-branch momentum projector_instance
        #######
        if self.instance_loss_weight > 0.0 and self.instance_loss_func not in [
            "RotPred",
            "DistAug_SimCLR",
            "SimCLR",
        ]:
            for param_q, param_k in zip(
                self.projector_instance.parameters(),
                self.projector_instance_k.parameters(),
            ):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                    1.0 - _contrast_momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue_ins(self, keys):
        # gather keys before updating instance branch queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.ins_queue_ptr)
        assert self.ins_queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.ins_queue[:, ptr : ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.ins_queue_len  # move pointer

        self.ins_queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_ins_target(self, keys):
        # gather keys before updating instance branch queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.ins_queue_target_ptr)
        assert self.ins_queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.ins_queue_target[ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.ins_queue_len  # move pointer

        self.ins_queue_target_ptr[0] = ptr

    def featprop(self, feat):
        N, C, H, W = feat.shape

        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)

        # Similarity calculation
        feat = F.normalize(feat, dim=1)

        # [N, C, H * W]
        feat = feat.view(N, C, -1)

        # [N, H * W, H * W]
        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.0:
            attention = attention + 1e-6
        attention = attention**self.pixpro_p

        # [N, C, H * W]
        feat = torch.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, H, W)

    def forward(self, *data):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if self.instance_branch_class_aware or (
            self.instance_loss_weight > 0.0
            and self.instance_loss_func in ["SimCLR", "MOCOv2"]
        ):
            assert len(data) == 5, DATA_INCOMPATIBLE_ERR
            (im_1, im_2, coord1, coord2, target) = data
        elif self.instance_loss_weight > 0.0 and self.instance_loss_func == "RotPred":
            assert len(data) == 8, DATA_INCOMPATIBLE_ERR
            (im_1, im_2, coord1, coord2, im_1_rot, im_2_rot, deg1, deg2) = data
        else:
            assert len(data) == 4, DATA_INCOMPATIBLE_ERR
            (im_1, im_2, coord1, coord2) = data

        ################
        # get features from encoder
        ################
        feat_1_c4, feat_1_c5 = self.encoder(
            im_1
        )  # feat1_c4.shape: (bs,256,14,14), feat1_c5.shape: (bs,512,7,7)
        feat_2_c4, feat_2_c5 = self.encoder(im_2)

        ################
        # get dense-branch features from projector g
        ################
        if self.dense_loss_weight > 0.0:
            proj_1_c4 = self.projector_c4(
                feat_1_c4
            )  # shape: (bs,256,14,14), NOT NORMALIZED
            proj_2_c4 = self.projector_c4(feat_2_c4)
            proj_1_c5 = self.projector_c5(
                feat_1_c5
            )  # shape: (bs,256,7,7), NOT NORMALIZED
            proj_2_c5 = self.projector_c5(feat_2_c5)

        ################
        # get dense-branch features from predictor
        ################
        if self.dense_loss_weight > 0.0 and self.dense_loss_func in ["DS2"]:
            pred_1_c4 = self.featprop(proj_1_c4)  # shape: (bs,256,14,14)
            pred_1_c4 = F.normalize(pred_1_c4, dim=1)  # shape: (bs,256,14,14)
            pred_2_c4 = self.featprop(
                proj_2_c4,
            )
            pred_2_c4 = F.normalize(pred_2_c4, dim=1)
            pred_1_c5 = self.featprop(proj_1_c5)  # shape: (bs,256,14,14)
            pred_1_c5 = F.normalize(pred_1_c5, dim=1)  # shape: (bs,256,14,14)
            pred_2_c5 = self.featprop(
                proj_2_c5,
            )
            pred_2_c5 = F.normalize(pred_2_c5, dim=1)

        ################
        # get instance-branch features from instance branch predictor g
        ################
        if self.instance_loss_weight > 0.0:
            if self.instance_loss_func == "RotPred":
                feat_1_rot = self.rot_MLP2d(feat_1_rot)  # shape: (bs, c_reduced, 7, 7)
                feat_2_rot = self.rot_MLP2d(feat_2_rot)

                feat_1_rot = feat_1_rot.view(
                    feat_1_rot.size(0), -1
                )  # flatten tensor, shape: (bs, c_reduced*49)
                feat_2_rot = feat_2_rot.view(feat_2_rot.size(0), -1)

                img1_rot_prediction = self.rot_predictor(
                    torch.cat(
                        (self.rot_MLP2d(feat_1).view(feat_1.size(0), -1), feat_1_rot),
                        dim=1,
                    )
                )  # shape: (bs, 4), 4 is total num of possible rotation multipliers
                img2_rot_prediction = self.rot_predictor(
                    torch.cat(
                        (self.rot_MLP2d(feat_2).view(feat_2.size(0), -1), feat_2_rot),
                        dim=1,
                    )
                )
                img1_rot_prediction = F.normalize(img1_rot_prediction, dim=1)
                img2_rot_prediction = F.normalize(img2_rot_prediction, dim=1)
            elif self.instance_loss_func == "DistAug_SimCLR":
                proj_instance_1 = self.fc_mlp(
                    self.avgpool(feat_1).view(feat_1.size(0), -1)
                )  # shape: (bs,128)
                proj_instance_2 = self.fc_mlp(
                    self.avgpool(feat_2).view(feat_2.size(0), -1)
                )  # shape: (bs,128)

                proj_instance_1 = F.normalize(proj_instance_1, dim=1)
                proj_instance_2 = F.normalize(proj_instance_2, dim=1)
            else:
                # instance_loss_func in ["Cosine", "DistAug_MOCOv2", "MOCOv2", "SimCLR"]

                proj_instance_1 = self.projector_instance(feat_1)
                proj_instance_2 = self.projector_instance(feat_2)

                if self.instance_loss_func == "SimCLR":
                    proj_instance_1 = F.normalize(
                        self.avgpool(proj_instance_1).view(proj_instance_1.size(0), -1),
                        dim=1,
                    )
                    proj_instance_2 = F.normalize(
                        self.avgpool(proj_instance_2).view(proj_instance_2.size(0), -1),
                        dim=1,
                    )
                elif self.instance_loss_func == "Cosine":
                    pred_instance_1 = self.predictor(proj_instance_1)
                    pred_instance_1 = F.normalize(
                        self.avgpool(pred_instance_1).view(pred_instance_1.size(0), -1),
                        dim=1,
                    )
                    pred_instance_2 = self.predictor(proj_instance_2)
                    pred_instance_2 = F.normalize(
                        self.avgpool(pred_instance_2).view(pred_instance_2.size(0), -1),
                        dim=1,
                    )
                elif self.instance_loss_func in ["DistAug_MOCOv2", "MOCOv2"]:
                    proj_instance_1 = F.normalize(
                        self.avgpool(proj_instance_1).view(proj_instance_1.size(0), -1),
                        dim=1,
                    )  # shape: (bs, C)
                    proj_instance_2 = F.normalize(
                        self.avgpool(proj_instance_2).view(proj_instance_2.size(0), -1),
                        dim=1,
                    )  # shape: (bs, C)

        # compute key (momentum) features
        with torch.no_grad():  # no gradient to keys
            # first, update the momentum encoder/projector/projector_instance's parameters
            self._momentum_update_key_encoder()

            if self.dense_loss_weight > 0.0:
                feat_1_ng_c4, feat_1_ng_c5 = self.encoder_k(im_1)
                feat_2_ng_c4, feat_2_ng_c5 = self.encoder_k(im_2)
            elif not (
                self.dense_loss_weight == 0.0
                and self.instance_loss_weight > 0.0
                and self.instance_loss_func in ["DistAug_SimCLR", "SimCLR", "RotPred"]
            ):
                feat_1_ng = self.encoder_k(im_1)
                feat_2_ng = self.encoder_k(im_2)

            if self.dense_loss_weight > 0.0 and self.dense_loss_func in ["DS2"]:
                proj_1_ng_c4 = self.projector_k_c4(feat_1_ng_c4)
                proj_2_ng_c4 = self.projector_k_c4(feat_2_ng_c4)
                proj_1_ng_c5 = self.projector_k_c5(feat_1_ng_c5)
                proj_2_ng_c5 = self.projector_k_c5(feat_2_ng_c5)
                proj_1_ng_c4 = F.normalize(proj_1_ng_c4, dim=1)
                proj_2_ng_c4 = F.normalize(proj_2_ng_c4, dim=1)
                proj_1_ng_c5 = F.normalize(proj_1_ng_c5, dim=1)
                proj_2_ng_c5 = F.normalize(proj_2_ng_c5, dim=1)

            if self.instance_loss_weight > 0.0 and self.instance_loss_func not in [
                "RotPred",
                "DistAug_SimCLR",
                "SimCLR",
            ]:
                proj_instance_1_ng = self.projector_instance_k(feat_1_ng)
                proj_instance_1_ng = F.normalize(
                    self.avgpool(proj_instance_1_ng).view(
                        proj_instance_1_ng.size(0), -1
                    ),
                    dim=1,
                )  # shape: (bs, C)

                proj_instance_2_ng = self.projector_instance_k(feat_2_ng)
                proj_instance_2_ng = F.normalize(
                    self.avgpool(proj_instance_2_ng).view(
                        proj_instance_2_ng.size(0), -1
                    ),
                    dim=1,
                )  # shape: (bs, C)

        ############
        # compute loss
        ############

        loss_dense = 0.0
        loss_instance = 0.0

        if self.dense_loss_weight > 0.0:
            loss_dense = 0.5 * (
                ds2_loss(
                    pred_1_c4,
                    proj_2_ng_c4,
                    feat_1_c4,
                    feat_2_ng_c4,
                    coord1,
                    coord2,
                )
                + ds2_loss(
                    pred_2_c4,
                    proj_1_ng_c4,
                    feat_2_c4,
                    feat_1_ng_c4,
                    coord2,
                    coord1,
                )
            ) + 0.5 * (
                ds2_loss(
                    pred_1_c5,
                    proj_2_ng_c5,
                    feat_1_c5,
                    feat_2_ng_c5,
                    coord1,
                    coord2,
                )
                + ds2_loss(
                    pred_2_c5,
                    proj_1_ng_c5,
                    feat_2_c5,
                    feat_1_ng_c5,
                    coord2,
                    coord1,
                )
            )

        if self.instance_loss_weight > 0.0:
            if self.instance_loss_func in ["SimCLR", "DistAug_SimCLR"]:
                loss_instance = simclr_loss(proj_instance_1, proj_instance_2)
            elif self.instance_loss_func == "Cosine":
                loss_instance = cosine_loss(
                    pred_instance_1, proj_instance_2_ng
                ) + cosine_loss(pred_instance_2, proj_instance_1_ng)
            elif self.instance_loss_func in ["DistAug_MOCOv2", "MOCOv2"]:
                loss_instance = distaug_mocov2_loss(
                    proj_instance_1,
                    proj_instance_2_ng,
                    self.ins_queue,
                    self.instance_branch_class_aware,
                    ins_queue_target=self.ins_queue_target
                    if self.instance_branch_class_aware
                    else None,
                    target=target if self.instance_branch_class_aware else None,
                ) + distaug_mocov2_loss(
                    proj_instance_2,
                    proj_instance_1_ng,
                    self.ins_queue,
                    self.instance_branch_class_aware,
                    ins_queue_target=self.ins_queue_target
                    if self.instance_branch_class_aware
                    else None,
                    target=target if self.instance_branch_class_aware else None,
                )
                # enqueue and dequeue
                self._dequeue_and_enqueue_ins(proj_instance_1_ng)
                if self.instance_branch_class_aware:
                    self._dequeue_and_enqueue_ins_target(target)
            elif self.instance_loss_func == "RotPred":
                loss_instance = rotpred_loss(img1_rot_prediction, deg1) + rotpred_loss(
                    img2_rot_prediction, deg2
                )

        return (
            self.dense_loss_weight * loss_dense
            + self.instance_loss_weight * loss_instance
        )


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
