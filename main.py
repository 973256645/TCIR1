import os
import time
import torch
import argparse
import csv
import numpy as np

from model import SASRec
from utils import *
from full_eval_adapter import evaluate_valid_full, evaluate_full, evaluate_split_full
from ce_train_adapter import compute_ce_loss
from bucket_eval_adapter import (
    format_bucket_metrics,
    load_item_bucket,
    resolve_bucket_path,
)

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def _overall_metric(metrics, name):
    # New format: (NDCG@5, HR@5, NDCG@10, HR@10).
    # Old two-value format is treated as (NDCG@10, HR@10) for defensive compatibility.
    if len(metrics) >= 4:
        index = {'ndcg5': 0, 'hr5': 1, 'ndcg10': 2, 'hr10': 3}[name]
        return metrics[index]
    if len(metrics) == 2:
        if name == 'ndcg10':
            return metrics[0]
        if name == 'hr10':
            return metrics[1]
    return 0.0


def format_overall_metrics(split, metrics):
    return '%s (N@5: %.4f, H@5: %.4f, N@10: %.4f, H@10: %.4f)' % (
        split,
        _overall_metric(metrics, 'ndcg5'),
        _overall_metric(metrics, 'hr5'),
        _overall_metric(metrics, 'ndcg10'),
        _overall_metric(metrics, 'hr10'),
    )


def _candidate_diagnostic_dirs(dataset):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join('diagnostics', dataset),
        os.path.join(script_dir, 'diagnostics', dataset),
        os.path.join('.', 'diagnostics', dataset),
    ]


def _resolve_tcir_bucket_path(args):
    if args.item_bucket_path is not None:
        return args.item_bucket_path
    if args.bucket_path is not None:
        return args.bucket_path
    for directory in _candidate_diagnostic_dirs(args.dataset):
        candidate = os.path.join(directory, 'item_bucket.npy')
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_tcir_popularity_path(args):
    if args.item_popularity_path is not None:
        return args.item_popularity_path
    for directory in _candidate_diagnostic_dirs(args.dataset):
        for filename in ['item_popularity.npy', 'item_popularity.csv']:
            candidate = os.path.join(directory, filename)
            if os.path.isfile(candidate):
                return candidate
    return None


def _load_tcir_item_bucket(bucket_path, itemnum):
    if bucket_path is None or not os.path.isfile(bucket_path):
        raise FileNotFoundError('TCIR requires item_bucket.npy, but it was not found.')
    item_bucket = np.asarray(np.load(bucket_path), dtype=np.int64).reshape(-1)
    if len(item_bucket) < itemnum + 1:
        raise ValueError('item_bucket.npy length must be >= itemnum + 1.')
    unique_values = set(item_bucket.tolist())
    if not unique_values.issubset({0, 1, 2, 3}):
        raise ValueError('item_bucket.npy values must be in {0, 1, 2, 3}.')
    if item_bucket[0] != 0:
        raise ValueError('item_bucket[0] must be 0 for padding / unknown.')
    return item_bucket[:itemnum + 1]


def _load_popularity_csv(popularity_path, itemnum):
    popularity = np.zeros(itemnum + 1, dtype=np.float32)
    with open(popularity_path, 'r', encoding='utf-8') as f_pop:
        reader = csv.DictReader(f_pop)
        if reader.fieldnames is None or 'item_id' not in reader.fieldnames or 'popularity' not in reader.fieldnames:
            raise ValueError('item_popularity.csv must contain item_id and popularity columns.')
        for row in reader:
            item_id = int(float(row['item_id']))
            if 1 <= item_id <= itemnum:
                popularity[item_id] = float(row['popularity'])
    return popularity


def _load_tcir_item_popularity(popularity_path, itemnum):
    if popularity_path is None or not os.path.isfile(popularity_path):
        raise FileNotFoundError('TCIR requires item_popularity.npy or item_popularity.csv, but it was not found.')
    if popularity_path.endswith('.npy'):
        popularity = np.asarray(np.load(popularity_path), dtype=np.float32).reshape(-1)
        if len(popularity) < itemnum + 1:
            raise ValueError('item_popularity.npy length must be >= itemnum + 1.')
        return popularity[:itemnum + 1]
    if popularity_path.endswith('.csv'):
        return _load_popularity_csv(popularity_path, itemnum)
    raise ValueError('item_popularity_path must point to .npy or .csv.')


def load_tcir_metadata(args, itemnum):
    bucket_path = _resolve_tcir_bucket_path(args)
    popularity_path = _resolve_tcir_popularity_path(args)
    item_bucket = _load_tcir_item_bucket(bucket_path, itemnum)
    item_popularity = _load_tcir_item_popularity(popularity_path, itemnum)
    print('TCIR metadata loaded: bucket=%s popularity=%s' % (bucket_path, popularity_path))
    return item_bucket, item_popularity


def _aux_value(aux, key):
    if aux is None or key not in aux:
        return 0.0
    value = aux[key]
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


TCIR_EPOCH_FIELDS = [
    'rho_mean',
    'rho_std',
    'rho_min',
    'rho_max',
    'r_mean',
    'r_std',
    'r_min',
    'r_max',
    'valid_proto_ratio',
    'tail_position_ratio',
    'rho_reg_loss',
    'tail_recon_loss',
    'time_nonzero_ratio',
    'log_gap_mean',
    'log_gap_var',
]

TCIR_DEBUG_FIELDS = [
    'seq_time_nonzero_ratio',
    'seq_time_min',
    'seq_time_max',
    'timestamps_is_none',
]

TCIR_POSITION_FIELDS = [
    'epoch',
    'user_id',
    'pos',
    'item_id',
    'timestamp',
    'bucket',
    'popularity',
    'eta',
    'log_gap_prev',
    'log_gap_next',
    'rho',
    'r',
    'valid_proto',
    'is_tail',
]


def init_tcir_epoch_meter():
    fields = TCIR_EPOCH_FIELDS + TCIR_DEBUG_FIELDS
    return {'count': 0, 'sums': {field: 0.0 for field in fields}}


def _tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_tcir_position_stats(rows, epoch, user_ids, aux, max_rows):
    if len(rows) >= max_rows:
        return
    required = [
        'seq_items',
        'timestamps',
        'bucket_ids',
        'popularity',
        'eta',
        'log_gap_prev',
        'log_gap_next',
        'rho',
        'r',
        'valid_proto_mask',
        'tail_pos_mask',
        'valid_mask',
    ]
    if aux is None or any(field not in aux for field in required):
        return

    user_ids = np.asarray(user_ids)
    values = {field: _tensor_to_numpy(aux[field]) for field in required}
    valid_mask = values['valid_mask'].astype(bool)
    valid_proto_mask = values['valid_proto_mask'].astype(bool)
    tail_pos_mask = values['tail_pos_mask'].astype(bool)

    for row_idx in range(valid_mask.shape[0]):
        user_id = int(user_ids[row_idx]) if row_idx < len(user_ids) else -1
        for pos_idx in range(valid_mask.shape[1]):
            if not valid_mask[row_idx, pos_idx]:
                continue
            rows.append({
                'epoch': int(epoch),
                'user_id': user_id,
                'pos': int(pos_idx),
                'item_id': int(values['seq_items'][row_idx, pos_idx]),
                'timestamp': float(values['timestamps'][row_idx, pos_idx]),
                'bucket': int(values['bucket_ids'][row_idx, pos_idx]),
                'popularity': float(values['popularity'][row_idx, pos_idx]),
                'eta': float(values['eta'][row_idx, pos_idx]),
                'log_gap_prev': float(values['log_gap_prev'][row_idx, pos_idx]),
                'log_gap_next': float(values['log_gap_next'][row_idx, pos_idx]),
                'rho': float(values['rho'][row_idx, pos_idx]),
                'r': float(values['r'][row_idx, pos_idx]),
                'valid_proto': int(valid_proto_mask[row_idx, pos_idx]),
                'is_tail': int(tail_pos_mask[row_idx, pos_idx]),
            })
            if len(rows) >= max_rows:
                return


def write_tcir_position_stats(run_dir, epoch, rows):
    path = os.path.join(run_dir, 'tcir_position_stats_epoch%d.csv' % epoch)
    with open(path, 'w', newline='', encoding='utf-8') as f_pos:
        writer = csv.DictWriter(f_pos, fieldnames=TCIR_POSITION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _seq_time_debug_stats(seq, seq_time):
    if seq_time is None:
        return {
            'seq_time_nonzero_ratio': 0.0,
            'seq_time_min': 0.0,
            'seq_time_max': 0.0,
        }
    seq_arr = np.asarray(seq)
    time_arr = np.asarray(seq_time)
    if seq_arr.shape != time_arr.shape:
        raise ValueError('seq and seq_time shapes differ: %s vs %s' % (seq_arr.shape, time_arr.shape))
    padding_time_sum = np.abs(time_arr[seq_arr == 0]).sum()
    if padding_time_sum > 0:
        raise ValueError('seq_time padding positions must be 0.')
    valid_mask = seq_arr != 0
    valid_count = valid_mask.sum()
    if valid_count == 0:
        return {
            'seq_time_nonzero_ratio': 0.0,
            'seq_time_min': 0.0,
            'seq_time_max': 0.0,
        }
    valid_time = time_arr[valid_mask]
    return {
        'seq_time_nonzero_ratio': float((valid_time > 0).sum() / valid_count),
        'seq_time_min': float(time_arr.min()),
        'seq_time_max': float(time_arr.max()),
    }


def update_tcir_epoch_meter(meter, aux, seq=None, seq_time=None):
    meter['count'] += 1
    for field in TCIR_EPOCH_FIELDS:
        meter['sums'][field] += _aux_value(aux, field)
    for field, value in _seq_time_debug_stats(seq, seq_time).items():
        meter['sums'][field] += value
    meter['sums']['timestamps_is_none'] += _aux_value(aux, 'timestamps_is_none')


def average_tcir_epoch_meter(meter):
    count = meter['count']
    if count == 0:
        return {field: 0.0 for field in TCIR_EPOCH_FIELDS + TCIR_DEBUG_FIELDS}
    return {field: meter['sums'][field] / count for field in TCIR_EPOCH_FIELDS + TCIR_DEBUG_FIELDS}


def format_tcir_time_debug(stats):
    return (
        'TCIR time debug: seq_time_nonzero_ratio=%.6f, seq_time_min=%.6f, seq_time_max=%.6f, '
        'timestamps_is_none=%.6f, tcir_time_nonzero_ratio=%.6f, log_gap_mean=%.6f, log_gap_var=%.6f'
        % (
            stats.get('seq_time_nonzero_ratio', 0.0),
            stats.get('seq_time_min', 0.0),
            stats.get('seq_time_max', 0.0),
            stats.get('timestamps_is_none', 0.0),
            stats.get('time_nonzero_ratio', 0.0),
            stats.get('log_gap_mean', 0.0),
            stats.get('log_gap_var', 0.0),
        )
    )


def format_tcir_train_log(aux):
    return (
        'rho_reg_loss: %.6f, tail_recon_loss: %.6f, '
        'rho_mean: %.4f, rho_std: %.4f, r_mean: %.4f, r_std: %.4f, '
        'valid_proto_ratio: %.4f, tail_position_ratio: %.4f'
        % (
            _aux_value(aux, 'rho_reg_loss'),
            _aux_value(aux, 'tail_recon_loss'),
            _aux_value(aux, 'rho_mean'),
            _aux_value(aux, 'rho_std'),
            _aux_value(aux, 'r_mean'),
            _aux_value(aux, 'r_std'),
            _aux_value(aux, 'valid_proto_ratio'),
            _aux_value(aux, 'tail_position_ratio'),
        )
    )


def write_tcir_eval_log(tcir_log_file, epoch, stats):
    values = [str(epoch)]
    values.extend('%.8f' % stats.get(field, 0.0) for field in TCIR_EPOCH_FIELDS)
    tcir_log_file.write('\t'.join(values) + '\n')
    tcir_log_file.flush()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--model_name', default='sasrec', type=str, choices=['sasrec', 'tcir_minimal'])
parser.add_argument('--bucket_eval', default=True, type=str2bool)
parser.add_argument('--bucket_path', default=None, type=str)
parser.add_argument('--item_bucket_path', default=None, type=str)
parser.add_argument('--item_popularity_path', default=None, type=str)
parser.add_argument('--tcir_proto_window', default=5, type=int)
parser.add_argument('--tcir_time_encoder_dim', default=16, type=int)
parser.add_argument('--tcir_pop_bucket_dim', default=8, type=int)
parser.add_argument('--tcir_reliability_hidden_dim', default=128, type=int)
parser.add_argument('--tcir_gate_hidden_dim', default=32, type=int)
parser.add_argument('--tcir_lambda_reg', default=0.01, type=float)
parser.add_argument('--tcir_lambda_tail', default=0.05, type=float)
parser.add_argument('--tcir_pi_rho', default=0.35, type=float)
parser.add_argument('--tcir_nu_rho', default=0.02, type=float)
parser.add_argument('--tcir_detach_reliability_input', default=True, type=str2bool)
parser.add_argument('--disable_tcir_eval', default=False, type=str2bool)
parser.add_argument('--disable_time_feature', default=False, type=str2bool)
# tcir_fast_eval true for no aux
parser.add_argument('--tcir_fast_eval', default=False, type=str2bool)
parser.add_argument('--save_tcir_position_stats', default=False, type=str2bool)
parser.add_argument('--position_stats_max_rows', default=500000, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':

    u2i_index, i2u_index = build_index(args.dataset)
    
    # global dataset
    if args.model_name == 'tcir_minimal':
        dataset = data_partition_with_time(args.dataset)
        [
            user_train,
            user_valid,
            user_test,
            user_train_time,
            user_valid_time,
            user_test_time,
            usernum,
            itemnum,
        ] = dataset
    else:
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset

    tcir_item_bucket = None
    tcir_item_popularity = None
    if args.model_name == 'tcir_minimal':
        tcir_item_bucket, tcir_item_popularity = load_tcir_metadata(args, itemnum)

    item_bucket = None
    bucket_eval_enabled = False
    bucket_path = resolve_bucket_path(args)
    try:
        if args.bucket_eval and args.model_name == 'tcir_minimal':
            item_bucket = tcir_item_bucket
            bucket_eval_enabled = True
            print('bucket evaluation enabled with TCIR item_bucket metadata')
        elif args.bucket_eval and os.path.isfile(bucket_path):
            item_bucket = load_item_bucket(bucket_path, itemnum)
            bucket_eval_enabled = True
            print('bucket evaluation enabled: %s' % bucket_path)
        elif not args.bucket_eval:
            print('bucket evaluation disabled by --bucket_eval false')
        else:
            print('bucket evaluation skipped: item_bucket.npy not found at %s' % bucket_path)
    except Exception as e:
        print('bucket evaluation skipped: failed to load item_bucket.npy (%s)' % str(e))

    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    log_header = 'epoch valid(N@5,H@5,N@10,H@10) test(N@5,H@5,N@10,H@10)'
    if bucket_eval_enabled:
        log_header += ' valid_bucket test_bucket'
    f.write(log_header + '\n')

    tcir_log_file = None
    if args.model_name == 'tcir_minimal':
        tcir_log_file = open(os.path.join(args.dataset + '_' + args.train_dir, 'tcir_log.txt'), 'w')
        tcir_log_file.write('epoch\t' + '\t'.join(TCIR_EPOCH_FIELDS) + '\n')
    tcir_epoch_meter = init_tcir_epoch_meter()
    tcir_position_rows = []
    
    if args.model_name == 'tcir_minimal':
        sampler = TimeAwareWarpSampler(
            user_train,
            user_train_time,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            n_workers=3,
        )
    else:
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    if args.model_name == 'sasrec':
        model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    else:
        from TCIR import TCIRMinimalSASRec
        model = TCIRMinimalSASRec(
            usernum,
            itemnum,
            args,
            item_bucket=tcir_item_bucket,
            item_popularity=tcir_item_popularity,
        ).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        if bucket_eval_enabled:
            t_test, t_test_bucket = evaluate_split_full(
                model, dataset, args, split='test', item_bucket=item_bucket, return_bucket=True
            )
            print(format_overall_metrics('test', t_test))
            print(format_bucket_metrics(t_test_bucket))
        else:
            t_test = evaluate_full(model, dataset, args)
            print(format_overall_metrics('test', t_test))
    
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            if args.model_name == 'tcir_minimal':
                u, seq, seq_time, pos, neg = sampler.next_batch() # tuples to ndarray
                u, seq, seq_time, pos, neg = (
                    np.array(u),
                    np.array(seq),
                    np.array(seq_time),
                    np.array(pos),
                    np.array(neg),
                )
            else:
                u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            adam_optimizer.zero_grad()
            if args.model_name == 'tcir_minimal':
                rec_loss = compute_ce_loss(model, seq, pos, args, seq_time=seq_time)
            else:
                rec_loss = compute_ce_loss(model, seq, pos, args)
            loss = rec_loss
            tcir_aux = None
            if args.model_name == 'tcir_minimal':
                tcir_aux = getattr(model, 'last_tcir_aux', None)
                if tcir_aux is None:
                    raise RuntimeError('TCIR auxiliary outputs were not produced during training.')
                loss = (
                    rec_loss
                    + args.tcir_lambda_reg * tcir_aux['rho_reg_loss']
                    + args.tcir_lambda_tail * tcir_aux['tail_recon_loss']
                )
            # torch.norm(param) returns the square root of the sum of squared weights (‖w‖₂), 
            loss.backward()
            adam_optimizer.step()
            if args.model_name == 'tcir_minimal':
                update_tcir_epoch_meter(tcir_epoch_meter, tcir_aux, seq=seq, seq_time=seq_time)
                if args.save_tcir_position_stats:
                    collect_tcir_position_stats(
                        tcir_position_rows,
                        epoch,
                        u,
                        tcir_aux,
                        args.position_stats_max_rows,
                    )
            # if args.model_name == 'tcir_minimal':
            #     print(
            #         "loss in epoch {} iteration {}: total {:.6f}, rec_loss {:.6f}, {}".format(
            #             epoch,
            #             step,
            #             loss.item(),
            #             rec_loss.item(),
            #             format_tcir_train_log(tcir_aux),
            #         )
            #     )
            # else:
            #     print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            if bucket_eval_enabled:
                t_test, t_test_bucket = evaluate_split_full(
                    model, dataset, args, split='test', item_bucket=item_bucket, return_bucket=True
                )
                t_valid, t_valid_bucket = evaluate_split_full(
                    model, dataset, args, split='valid', item_bucket=item_bucket, return_bucket=True
                )
            else:
                t_test = evaluate_full(model, dataset, args)
                t_valid = evaluate_valid_full(model, dataset, args)
            valid_overall_str = format_overall_metrics('valid', t_valid)
            test_overall_str = format_overall_metrics('test', t_test)
            print('epoch:%d, time: %f(s), %s, %s'
                    % (epoch, T, valid_overall_str, test_overall_str))
            bucket_log = ''
            if bucket_eval_enabled:
                valid_bucket_str = format_bucket_metrics(t_valid_bucket)
                test_bucket_str = format_bucket_metrics(t_test_bucket)
                print(valid_bucket_str)
                print(test_bucket_str)
                bucket_log = ' ' + valid_bucket_str + ' ' + test_bucket_str

            if args.model_name == 'tcir_minimal' and tcir_log_file is not None:
                tcir_epoch_stats = average_tcir_epoch_meter(tcir_epoch_meter)
                write_tcir_eval_log(tcir_log_file, epoch, tcir_epoch_stats)
                print(format_tcir_time_debug(tcir_epoch_stats))
                tcir_epoch_meter = init_tcir_epoch_meter()
                if args.save_tcir_position_stats:
                    position_path = write_tcir_position_stats(
                        args.dataset + '_' + args.train_dir,
                        epoch,
                        tcir_position_rows,
                    )
                    print('TCIR position stats written: %s (%d rows)' % (position_path, len(tcir_position_rows)))
                    tcir_position_rows = []

            valid_ndcg10 = _overall_metric(t_valid, 'ndcg10')
            valid_hr10 = _overall_metric(t_valid, 'hr10')
            test_ndcg10 = _overall_metric(t_test, 'ndcg10')
            test_hr10 = _overall_metric(t_test, 'hr10')
            if valid_ndcg10 > best_val_ndcg or valid_hr10 > best_val_hr or test_ndcg10 > best_test_ndcg or test_hr10 > best_test_hr:
                best_val_ndcg = max(valid_ndcg10, best_val_ndcg)
                best_val_hr = max(valid_hr10, best_val_hr)
                best_test_ndcg = max(test_ndcg10, best_test_ndcg)
                best_test_hr = max(test_hr10, best_test_hr)
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(epoch) + ' ' + valid_overall_str + ' ' + test_overall_str + bucket_log + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    if tcir_log_file is not None:
        tcir_log_file.close()
    sampler.close()
    print("Done")
