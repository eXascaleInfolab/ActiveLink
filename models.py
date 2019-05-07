import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal


class ConvE(torch.nn.Module):
    def __init__(self, config, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.emb_e = torch.nn.Embedding(num_entities, config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(config.feature_map_dropout)
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc2 = torch.nn.Linear(10368, self.embedding_dim)

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def _embedding(self, input, weight):
        return F.embedding(input, weight.cuda())

    def _conv2d(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return F.conv2d(input, weight.cuda(), bias.cuda(), stride, padding, dilation, groups)

    def _batch_norm(self, num_features, input, weight=None, bias=None, training=True, eps=1e-5, momentum=0.1):
        running_mean = torch.zeros(num_features).cuda()
        running_var = torch.ones(num_features).cuda()
        return F.batch_norm(input, running_mean, running_var, weight.cuda(), bias.cuda(), training, momentum, eps)

    def _linear(self, input, weight, bias):
        return F.linear(input, weight.cuda(), bias.cuda())

    def forward(self, e1, rel, batch_size=None, weights=None):
        if batch_size is None:
            batch_size = self.batch_size

        if weights is None:
            e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
            rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)  # out: (128L, 1L, 20L, 20L)
            x = self.bn0(stacked_inputs)
            x = self.inp_drop(x)
            x = self.conv1(x)  # out: (128L, 32L, 18L, 18L)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)  # out: (128L, 10368L)
            x = self.fc2(x)  # out: (128L, 200L)
            x = self.hidden_drop(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = torch.mm(x, self.emb_e.weight.transpose(1, 0))  # out: (128L, 14541L)
            pred = x + self.b.expand_as(x)
        else:
            e1_embedded = self._embedding(e1, weights['emb_e.weight']).view(-1, 1, 10, 20)
            rel_embedded = self._embedding(rel, weights['emb_rel.weight']).view(-1, 1, 10, 20)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
            x = self._batch_norm(1, stacked_inputs, weights['bn0.weight'], weights['bn0.bias'])
            x = self.inp_drop(x)
            x = self._conv2d(x, weights['conv1.weight'], weights['conv1.bias'])
            x = self._batch_norm(32, x, weights['bn1.weight'], weights['bn1.bias'])
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self._linear(x, weights['fc2.weight'], weights['fc2.bias'])
            x = self.hidden_drop(x)
            x = self._batch_norm(self.embedding_dim, x, weights['bn2.weight'], weights['bn2.bias'])
            x = F.relu(x)
            x = torch.mm(x, weights['emb_e.weight'].transpose(1, 0))
            pred = x + weights['b'].expand_as(x)

        return pred

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, torch.nn.Linear) or isinstance(m_to, torch.nn.Conv2d) or isinstance(m_to, torch.nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
            if isinstance(m_to, torch.nn.Embedding):
                m_to.weight.data = m_from.weight.data.clone()


class MultilayerPerceptropn(torch.nn.Module):
    def __init__(self, config, num_entities, num_relations):
        super(MultilayerPerceptropn, self).__init__()
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.emb_e = torch.nn.Embedding(num_entities, config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(config.feature_map_dropout)
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc1 = torch.nn.Linear(400, self.embedding_dim)
        self.fc2 = torch.nn.Linear(200, self.embedding_dim)

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def _embedding(self, input, weight):
        return F.embedding(input, weight.cuda())

    def _batch_norm(self, num_features, input, weight=None, bias=None, training=True, eps=1e-5, momentum=0.1):
        running_mean = torch.zeros(num_features).cuda()
        running_var = torch.ones(num_features).cuda()
        return F.batch_norm(input, running_mean, running_var, weight.cuda(), bias.cuda(), training, momentum, eps)

    def _linear(self, input, weight, bias):
        return F.linear(input, weight.cuda(), bias.cuda())

    def forward(self, e1, rel, batch_size=None, weights=None):
        if batch_size is None:
            batch_size = self.batch_size

        if weights is None:
            e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
            rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
            x = self.bn0(stacked_inputs)
            x = self.inp_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = self.fc2(x)
            x = self.hidden_drop(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
            pred = x + self.b.expand_as(x)
        else:
            e1_embedded = self._embedding(e1, weights['emb_e.weight']).view(-1, 1, 10, 20)
            rel_embedded = self._embedding(rel, weights['emb_rel.weight']).view(-1, 1, 10, 20)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
            x = self._batch_norm(1, stacked_inputs, weights['bn0.weight'], weights['bn0.bias'])
            x = self.inp_drop(x)
            x = x.view(batch_size, -1)
            x = self._linear(x, weights['fc1.weight'], weights['fc1.bias'])
            x = self._batch_norm(self.embedding_dim, x, weights['bn1.weight'], weights['bn1.bias'])
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = self._linear(x, weights['fc2.weight'], weights['fc2.bias'])
            x = self.hidden_drop(x)
            x = self._batch_norm(self.embedding_dim, x, weights['bn2.weight'], weights['bn2.bias'])
            x = F.relu(x)
            x = torch.mm(x, weights['emb_e.weight'].transpose(1, 0))
            pred = x + weights['b'].expand_as(x)

        return pred
