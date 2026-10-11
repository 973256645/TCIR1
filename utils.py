import sys
import copy
import os
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt(resolve_dataset_file(dataset_name), dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index


def resolve_dataset_file(fname):
    candidates = [
        'data/%s.txt' % fname,
        'tcir_data/%s/%s.txt' % (fname, fname),
        os.path.join(os.path.dirname(__file__), 'data', '%s.txt' % fname),
        os.path.join(os.path.dirname(__file__), 'tcir_data', fname, '%s.txt' % fname),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def resolve_dataset_time_file(fname):
    candidates = [
        'data/%s_time.txt' % fname,
        'tcir_data/%s/%s_time.txt' % (fname, fname),
        os.path.join(os.path.dirname(__file__), 'data', '%s_time.txt' % fname),
        os.path.join(os.path.dirname(__file__), 'tcir_data', fname, '%s_time.txt' % fname),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def time_aware_sample_function(user_train, user_train_time, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq_time = np.zeros([maxlen], dtype=np.float32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        hist_items = user_train[uid][:-1]
        hist_times = user_train_time[uid][:-1]
        for i, t in zip(reversed(hist_items), reversed(hist_times)):
            seq[idx] = i
            seq_time[idx] = t
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, seq_time, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class TimeAwareWarpSampler(object):
    def __init__(self, User, UserTime, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=time_aware_sample_function, args=(User,
                                                                 UserTime,
                                                                 usernum,
                                                                 itemnum,
                                                                 batch_size,
                                                                 maxlen,
                                                                 self.result_queue,
                                                                 np.random.randint(2e9)
                                                                 )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(resolve_dataset_file(fname), 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition_with_time(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    UserTime = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_train_time = {}
    user_valid_time = {}
    user_test_time = {}

    data_path = resolve_dataset_file(fname)
    time_path = resolve_dataset_time_file(fname)

    with open(data_path, 'r') as f:
        data_lines = [line.rstrip() for line in f if line.strip()]

    if not os.path.isfile(time_path):
        print('[TimeData] warning: %s not found; using all-zero timestamps.' % time_path)
        time_lines = []
        for line in data_lines:
            u, i = line.split(' ')
            time_lines.append('%s %s 0' % (u, i))
    else:
        with open(time_path, 'r') as f:
            time_lines = [line.rstrip() for line in f if line.strip()]

    if len(data_lines) != len(time_lines):
        raise ValueError('%s and %s line counts differ.' % (data_path, time_path))

    for line_idx, (data_line, time_line) in enumerate(zip(data_lines, time_lines)):
        u, i = data_line.split(' ')
        tu, ti, timestamp = time_line.split(' ')
        if u != tu or i != ti:
            raise ValueError('data/time user-item mismatch at line %d.' % (line_idx + 1))
        u = int(u)
        i = int(i)
        try:
            timestamp = float(timestamp)
        except ValueError:
            raise ValueError('timestamp is not numeric at line %d.' % (line_idx + 1))
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        UserTime[u].append(timestamp)

    for user in User:
        if len(User[user]) != len(UserTime[user]):
            raise ValueError('item and timestamp sequence lengths differ for user %d.' % user)
        nfeedback = len(User[user])
        if nfeedback < 4:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
            user_train_time[user] = UserTime[user]
            user_valid_time[user] = []
            user_test_time[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
            user_train_time[user] = UserTime[user][:-2]
            user_valid_time[user] = [UserTime[user][-2]]
            user_test_time[user] = [UserTime[user][-1]]
    return [user_train, user_valid, user_test, user_train_time, user_valid_time, user_test_time, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
