import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F


from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.action_value.mplight import build_comp_mask, FRAP


def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv2d_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    pad_type="zero",
    activation=None,
):
    block = []
    assert pad_type in [
        "zero",
        "reflect",
        "replication",
    ], "invalid padding type: {}".format(pad_type)
    if pad_type == "zero":
        pass
    elif pad_type == "reflect":
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == "replication":
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    )
    xavier_normal_(block[-1].weight)
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def fc_block(
    in_channels,
    out_channels,
    activation=None,
    use_dropout=False,
    norm_type=None,
    dropout_probability=0.5,
):
    block = [nn.Linear(in_channels, out_channels)]
    xavier_normal_(block[-1].weight)
    if norm_type is not None and norm_type != "none":
        if norm_type == "LN":
            block.append(nn.LayerNorm(out_channels))
        else:
            raise NotImplementedError
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    elif activation is None:
        pass
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class ModelBody(nn.Module):
    def __init__(self, input_size, fc_layer_size, state_keys, device="cpu"):
        super(ModelBody, self).__init__()
        competition_mask = build_comp_mask(cfg.phase_pairs)

        self.frap = ModedFRAP(
            len(cfg.phase_pairs), cfg.phase_pairs, competition_mask.to(device), device
        ).to(device)

    def forward(self, states):
        x = self.frap(states)
        return x


class ModedFRAP(FRAP):
    def __init__(self, output_shape, phase_pairs, competition_mask, device):
        super(ModedFRAP, self).__init__(
            output_shape, phase_pairs, competition_mask, device
        )
        self.before_merge = nn.Conv2d(20, len(cfg.phase_pairs), kernel_size=(1, 1))

    def forward(self, states):
        states = states.to(self.device)
        num_movements = len(cfg.directions)
        batch_size = states.size()[0]
        states = states.float()

        acts = []
        for i in range(batch_size):
            phase_code = []
            for j in range(len(cfg.directions)):
                if states[i, j * cfg.demand_shape] == 1:
                    phase_code.append(j)
            if len(phase_code) == 0:
                acts = [0] * batch_size
                print(
                    "WARNING: This should only happen once. CoSLight doesn't record first observation until t+1."
                )
                break
            act_idx = None
            for i, pair in enumerate(cfg.phase_pairs):
                left = cfg.directions[pair[0]]
                right = cfg.directions[pair[1]]
                if (left in phase_code) and (right in phase_code):
                    act_idx = i
            acts.append(act_idx)

        # Expand action index to mark demand input indices
        extended_acts = []
        for i in range(batch_size):
            act_idx = acts[i]
            pair = self.phase_pairs[act_idx]
            zeros = torch.zeros(num_movements, dtype=torch.int64, device=self.device)
            zeros[cfg.directions[pair[0]]] = 1
            zeros[cfg.directions[pair[1]]] = 1
            extended_acts.append(zeros)
        extended_acts = torch.stack(extended_acts)
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        for i in range(num_movements):
            phase = phase_embeds[:, i]  # size 4
            demand_start = i * cfg.demand_shape
            demand_end = demand_start + cfg.demand_shape
            demand = states[:, demand_start:demand_end]
            demand = torch.sigmoid(self.d(demand))  # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

        pairs = []
        for pair in self.phase_pairs:
            pairs.append(
                phase_demands[:, cfg.directions[pair[0]]]
                + phase_demands[:, cfg.directions[pair[1]]]
            )

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j:
                    rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(
            rotated_phases,
            (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units),
        )
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(
            self.lane_conv(rotated_phases)
        )  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(
            self.hidden_layer(combine_features)
        )  # Phase competition representation
        combine_features = self.before_merge(
            combine_features
        )  # Pairwise competition result

        # Phase score
        # combine_features = torch.reshape(
        #     combine_features, (batch_size, self.oshape, self.oshape - 1)
        # )
        # q_values = torch.sum(combine_features, dim=-1)
        return torch.sum(combine_features, dim=-1)
