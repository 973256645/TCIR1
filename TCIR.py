import torch
import torch.nn.functional as F

from model import SASRec


class TCIRMinimalSASRec(SASRec):
    """SASRec with TCIR-minimal inserted after final hidden states.

    This class keeps the original SASRec backbone and prediction head intact.
    TCIR-minimal only reconstructs tail positions with a valid historical local
    prototype before the logits are computed.
    """

    def __init__(self, user_num, item_num, args, item_bucket, item_popularity):
        super(TCIRMinimalSASRec, self).__init__(user_num, item_num, args)

        self.tcir_proto_window = args.tcir_proto_window
        self.tcir_pi_rho = args.tcir_pi_rho
        self.tcir_nu_rho = args.tcir_nu_rho
        self.tcir_detach_reliability_input = args.tcir_detach_reliability_input
        self.disable_tcir_eval = args.disable_tcir_eval
        self.disable_time_feature = getattr(args, "disable_time_feature", False)
        self.tcir_fast_eval = getattr(args, "tcir_fast_eval", True)

        item_bucket = torch.as_tensor(item_bucket, dtype=torch.long)
        item_popularity = torch.as_tensor(item_popularity, dtype=torch.float32)
        self.register_buffer("tcir_item_bucket", item_bucket)
        self.register_buffer("tcir_item_popularity", item_popularity)
        max_log_pop = torch.log1p(item_popularity).max().clamp_min(1e-8)
        self.register_buffer("tcir_max_log_pop", max_log_pop)

        hidden_units = args.hidden_units
        self.tcir_time_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, args.tcir_time_encoder_dim),
            torch.nn.ReLU(),
        )
        self.tcir_bucket_emb = torch.nn.Embedding(4, args.tcir_pop_bucket_dim, padding_idx=0)

        reliability_input_dim = (
            hidden_units * 3
            + args.tcir_time_encoder_dim
            + args.tcir_pop_bucket_dim
            + 1
        )
        self.tcir_reliability_mlp = torch.nn.Sequential(
            torch.nn.Linear(reliability_input_dim, args.tcir_reliability_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.tcir_reliability_hidden_dim, 1),
        )
        self.tcir_gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, args.tcir_gate_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.tcir_gate_hidden_dim, 1),
        )
        self.last_tcir_aux = None

    def log2feats(self, log_seqs, timestamps=None):
        hidden = super(TCIRMinimalSASRec, self).log2feats(log_seqs)

        if (not self.training) and self.disable_tcir_eval:
            self.last_tcir_aux = self._zero_aux(hidden, log_seqs)
            return hidden

        seq_items = torch.LongTensor(log_seqs).to(hidden.device)
        timestamps_was_none = timestamps is None
        if timestamps_was_none:
            timestamps = torch.zeros_like(seq_items, dtype=torch.float32, device=hidden.device)
        else:
            timestamps = torch.as_tensor(timestamps, dtype=torch.float32, device=hidden.device)
        if self.disable_time_feature:
            timestamps = torch.zeros_like(timestamps)

        h_tilde, aux = self._apply_tcir_minimal(hidden, seq_items, timestamps, timestamps_was_none)
        self.last_tcir_aux = aux
        return h_tilde

    def _safe_item_lookup(self, values, seq_items):
        safe_items = seq_items.clamp(min=0, max=values.size(0) - 1)
        return values[safe_items]

    def _build_temporal_features(self, timestamps, valid_mask):
        bsz, seq_len = timestamps.shape
        if self.disable_time_feature:
            return torch.zeros((bsz, seq_len, 1), dtype=torch.float32, device=timestamps.device)
        gap_minus = torch.zeros((bsz, seq_len), dtype=torch.float32, device=timestamps.device)

        if seq_len > 1:
            prev_delta = timestamps[:, 1:] - timestamps[:, :-1]
            valid_prev = valid_mask[:, 1:] & valid_mask[:, :-1]
            gap_minus[:, 1:] = torch.where(valid_prev, prev_delta, torch.zeros_like(prev_delta))

        gap_minus = torch.log1p(torch.clamp(gap_minus, min=0.0))
        return gap_minus.unsqueeze(-1)

    def _build_log_gap_next_for_aux(self, timestamps, valid_mask):
        bsz, seq_len = timestamps.shape
        gap_next = torch.zeros((bsz, seq_len), dtype=torch.float32, device=timestamps.device)
        if self.disable_time_feature or seq_len <= 1:
            return gap_next
        next_delta = timestamps[:, 1:] - timestamps[:, :-1]
        valid_next = valid_mask[:, 1:] & valid_mask[:, :-1]
        gap_next[:, :-1] = torch.where(valid_next, next_delta, torch.zeros_like(next_delta))
        return torch.log1p(torch.clamp(gap_next, min=0.0))

    def _prefix_representation(self, hidden_for_input, valid_mask):
        valid_float = valid_mask.float().unsqueeze(-1)
        prefix_sum = torch.cumsum(hidden_for_input * valid_float, dim=1)
        prefix_count = torch.cumsum(valid_float, dim=1).clamp_min(1.0)
        return prefix_sum / prefix_count

    def _local_prototype_slow(self, hidden, valid_mask):
        bsz, seq_len, hidden_dim = hidden.shape
        source = hidden.detach()
        z_local = source.clone()
        valid_proto_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=hidden.device)

        for t in range(seq_len):
            start = max(0, t - self.tcir_proto_window)
            if start == t:
                continue
            hist = source[:, start:t, :]
            hist_mask = valid_mask[:, start:t]
            denom = hist_mask.float().sum(dim=1)
            has_proto = denom > 0
            if has_proto.any():
                proto = (hist * hist_mask.float().unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0).unsqueeze(-1)
                z_local[has_proto, t, :] = proto[has_proto]
                valid_proto_mask[has_proto, t] = True

        return z_local, valid_proto_mask

    def _local_prototype(self, hidden, valid_mask):
        bsz, seq_len, hidden_dim = hidden.shape
        source = hidden.detach()
        valid_proto_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=hidden.device)
        window = int(self.tcir_proto_window)
        if seq_len == 0 or window <= 0:
            return source.clone(), valid_proto_mask

        valid_float = valid_mask.float().unsqueeze(-1)
        masked_source = source * valid_float
        padded_source = torch.cat([source.new_zeros((bsz, window, hidden_dim)), masked_source], dim=1)
        padded_count = torch.cat([valid_float.new_zeros((bsz, window, 1)), valid_float], dim=1)

        window_sum = padded_source.unfold(1, window, 1)[:, :seq_len, :, :].sum(dim=-1)
        window_count = padded_count.unfold(1, window, 1)[:, :seq_len, :, :].sum(dim=-1)
        valid_proto_mask = window_count.squeeze(-1) > 0
        proto = window_sum / window_count.clamp_min(1.0)
        if proto.dtype != source.dtype:
            proto = proto.to(dtype=source.dtype)

        return torch.where(valid_proto_mask.unsqueeze(-1), proto, source), valid_proto_mask

    def _apply_tcir_minimal(self, hidden, seq_items, timestamps, timestamps_was_none):
        valid_mask = seq_items != 0
        bucket_ids = self._safe_item_lookup(self.tcir_item_bucket, seq_items)
        tail_pos_mask = (bucket_ids == 3) & valid_mask

        bsz, seq_len, hidden_dim = hidden.shape
        zeros_h = torch.zeros((bsz, 1, hidden_dim), dtype=hidden.dtype, device=hidden.device)
        h_prev = torch.cat([zeros_h, hidden[:, :-1, :]], dim=1)
        prev_valid = torch.cat(
            [torch.zeros((bsz, 1), dtype=torch.bool, device=hidden.device), valid_mask[:, :-1]],
            dim=1,
        )
        h_prev = h_prev * prev_valid.float().unsqueeze(-1)

        reliability_hidden = hidden.detach() if self.tcir_detach_reliability_input else hidden
        reliability_prev = h_prev.detach() if self.tcir_detach_reliability_input else h_prev
        q_prefix = self._prefix_representation(reliability_hidden, valid_mask)

        temporal_features = self._build_temporal_features(timestamps, valid_mask)
        time_features = self.tcir_time_encoder(temporal_features)
        bucket_features = self.tcir_bucket_emb(bucket_ids.clamp(min=0, max=3))
        tail_flag = tail_pos_mask.float().unsqueeze(-1)

        reliability_input = torch.cat(
            [reliability_prev, reliability_hidden, q_prefix, time_features, bucket_features, tail_flag],
            dim=-1,
        )
        rho = torch.sigmoid(self.tcir_reliability_mlp(reliability_input)).squeeze(-1)
        rho = rho * tail_pos_mask.float()

        z_local, valid_proto_mask = self._local_prototype(hidden, valid_mask)

        popularity = self._safe_item_lookup(self.tcir_item_popularity, seq_items)
        eta = 1.0 - torch.log1p(popularity) / self.tcir_max_log_pop
        eta = torch.clamp(eta, min=0.0, max=1.0)
        dist = torch.norm(hidden - z_local, p=2, dim=-1)

        gate_input = torch.stack([rho, eta, dist], dim=-1)
        r = torch.sigmoid(self.tcir_gate_mlp(gate_input)).squeeze(-1)
        recon_mask = tail_pos_mask & valid_proto_mask
        r = r * recon_mask.float()

        h_tilde = hidden + r.unsqueeze(-1) * (z_local - hidden)
        if (not self.training) and self.tcir_fast_eval:
            aux = self._build_fast_eval_aux(
                rho=rho,
                r=r,
                z_local=z_local,
                tail_pos_mask=tail_pos_mask,
                valid_proto_mask=valid_proto_mask,
                valid_mask=valid_mask,
                timestamps=timestamps,
                timestamps_was_none=timestamps_was_none,
                seq_items=seq_items,
                bucket_ids=bucket_ids,
                popularity=popularity,
                eta=eta,
                log_gap_prev=temporal_features[:, :, 0],
            )
            return h_tilde, aux

        rho_reg_loss, tail_recon_loss = self._aux_losses(
            hidden, h_tilde, z_local, rho, tail_pos_mask, recon_mask
        )

        # Original full diagnostic aux disabled during training for faster runs.
        aux = self._build_aux(
            rho=rho,
            r=r,
            z_local=z_local,
            tail_pos_mask=tail_pos_mask,
            valid_proto_mask=valid_proto_mask,
            rho_reg_loss=rho_reg_loss,
            tail_recon_loss=tail_recon_loss,
            valid_mask=valid_mask,
            recon_mask=recon_mask,
            timestamps=timestamps,
            timestamps_was_none=timestamps_was_none,
            seq_items=seq_items,
            bucket_ids=bucket_ids,
            popularity=popularity,
            eta=eta,
            log_gap_prev=temporal_features[:, :, 0],
            log_gap_next=self._build_log_gap_next_for_aux(timestamps, valid_mask),
        )
        # aux = {
        #     "rho_reg_loss": rho_reg_loss,
        #     "tail_recon_loss": tail_recon_loss,
        # }
        return h_tilde, aux

    def _aux_losses(self, hidden, h_tilde, z_local, rho, tail_pos_mask, recon_mask):
        zero = hidden.sum() * 0.0
        if tail_pos_mask.any():
            rho_tail = rho[tail_pos_mask]
            rho_reg_loss = (
                (rho_tail.mean() - self.tcir_pi_rho) ** 2
                + torch.relu(torch.as_tensor(self.tcir_nu_rho, dtype=hidden.dtype, device=hidden.device) - rho_tail.var(unbiased=False))
            )
        else:
            rho_reg_loss = zero

        if recon_mask.any():
            cosine = F.cosine_similarity(h_tilde[recon_mask], z_local.detach()[recon_mask], dim=-1, eps=1e-8)
            tail_recon_loss = torch.mean(rho[recon_mask].detach() * (1.0 - cosine))
        else:
            tail_recon_loss = zero
        return rho_reg_loss, tail_recon_loss

    def _stat_or_zero(self, values, mask, op):
        if mask.any():
            selected = values[mask]
            if op == "mean":
                return selected.mean().detach()
            if op == "std":
                return selected.std(unbiased=False).detach()
            if op == "min":
                return selected.min().detach()
            if op == "max":
                return selected.max().detach()
            if op == "var":
                return selected.var(unbiased=False).detach()
        return values.sum().detach() * 0.0

    def _build_aux(
        self,
        rho,
        r,
        z_local,
        tail_pos_mask,
        valid_proto_mask,
        rho_reg_loss,
        tail_recon_loss,
        valid_mask,
        recon_mask,
        timestamps,
        timestamps_was_none,
        seq_items,
        bucket_ids,
        popularity,
        eta,
        log_gap_prev,
        log_gap_next,
    ):
        valid_float = valid_mask.float()
        valid_count = valid_float.sum().clamp_min(1.0)
        tail_float = tail_pos_mask.float()
        proto_float = valid_proto_mask.float() * valid_float
        time_nonzero_ratio = (((timestamps > 0) & valid_mask).float().sum() / valid_count).detach()

        if timestamps.size(1) > 1:
            adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
            raw_gap = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0.0)
            log_gap = torch.log1p(raw_gap)
            log_gap_mean = self._stat_or_zero(log_gap, adjacent_valid, "mean")
            log_gap_var = self._stat_or_zero(log_gap, adjacent_valid, "var")
        else:
            zero = timestamps.sum().detach() * 0.0
            log_gap_mean = zero
            log_gap_var = zero

        aux = {
            "rho": rho,
            "r": r,
            "z_local": z_local,
            "tail_pos_mask": tail_pos_mask,
            "valid_proto_mask": valid_proto_mask,
            "seq_items": seq_items,
            "timestamps": timestamps,
            "bucket_ids": bucket_ids,
            "popularity": popularity,
            "eta": eta,
            "log_gap_prev": log_gap_prev,
            "log_gap_next": log_gap_next,
            "valid_mask": valid_mask,
            "rho_reg_loss": rho_reg_loss,
            "tail_recon_loss": tail_recon_loss,
            "rho_mean": self._stat_or_zero(rho, tail_pos_mask, "mean"),
            "rho_std": self._stat_or_zero(rho, tail_pos_mask, "std"),
            "rho_min": self._stat_or_zero(rho, tail_pos_mask, "min"),
            "rho_max": self._stat_or_zero(rho, tail_pos_mask, "max"),
            "r_mean": self._stat_or_zero(r, recon_mask, "mean"),
            "r_std": self._stat_or_zero(r, recon_mask, "std"),
            "r_min": self._stat_or_zero(r, recon_mask, "min"),
            "r_max": self._stat_or_zero(r, recon_mask, "max"),
            "valid_proto_ratio": (proto_float.sum() / valid_count).detach(),
            "tail_position_ratio": (tail_float.sum() / valid_count).detach(),
            "timestamps_is_none": torch.as_tensor(
                float(timestamps_was_none), dtype=torch.float32, device=timestamps.device
            ).detach(),
            "time_nonzero_ratio": time_nonzero_ratio,
            "log_gap_mean": log_gap_mean,
            "log_gap_var": log_gap_var,
        }
        return aux

    def _build_fast_eval_aux(
        self,
        rho,
        r,
        z_local,
        tail_pos_mask,
        valid_proto_mask,
        valid_mask,
        timestamps,
        timestamps_was_none,
        seq_items,
        bucket_ids,
        popularity,
        eta,
        log_gap_prev,
    ):
        zero_scalar = rho.sum().detach() * 0.0
        zero_loss = rho.sum() * 0.0
        return {
            "rho": rho,
            "r": r,
            "z_local": z_local,
            "tail_pos_mask": tail_pos_mask,
            "valid_proto_mask": valid_proto_mask,
            "seq_items": seq_items,
            "timestamps": timestamps,
            "bucket_ids": bucket_ids,
            "popularity": popularity,
            "eta": eta,
            "log_gap_prev": log_gap_prev,
            "log_gap_next": torch.zeros_like(log_gap_prev),
            "valid_mask": valid_mask,
            "rho_reg_loss": zero_loss,
            "tail_recon_loss": zero_loss,
            "rho_mean": zero_scalar,
            "rho_std": zero_scalar,
            "rho_min": zero_scalar,
            "rho_max": zero_scalar,
            "r_mean": zero_scalar,
            "r_std": zero_scalar,
            "r_min": zero_scalar,
            "r_max": zero_scalar,
            "valid_proto_ratio": zero_scalar,
            "tail_position_ratio": zero_scalar,
            "timestamps_is_none": torch.as_tensor(
                float(timestamps_was_none), dtype=torch.float32, device=timestamps.device
            ).detach(),
            "time_nonzero_ratio": zero_scalar,
            "log_gap_mean": zero_scalar,
            "log_gap_var": zero_scalar,
        }

    def _zero_aux(self, hidden, log_seqs):
        seq_items = torch.LongTensor(log_seqs).to(hidden.device)
        valid_mask = seq_items != 0
        zero_scalar = hidden.sum().detach() * 0.0
        zero_loss = hidden.sum() * 0.0
        rho = torch.zeros(seq_items.shape, dtype=hidden.dtype, device=hidden.device)
        r = torch.zeros_like(rho)
        z_local = hidden.detach()
        mask = torch.zeros(seq_items.shape, dtype=torch.bool, device=hidden.device)
        return {
            "rho": rho,
            "r": r,
            "z_local": z_local,
            "tail_pos_mask": mask,
            "valid_proto_mask": mask,
            "seq_items": seq_items,
            "timestamps": torch.zeros(seq_items.shape, dtype=hidden.dtype, device=hidden.device),
            "bucket_ids": torch.zeros(seq_items.shape, dtype=torch.long, device=hidden.device),
            "popularity": torch.zeros(seq_items.shape, dtype=hidden.dtype, device=hidden.device),
            "eta": torch.zeros(seq_items.shape, dtype=hidden.dtype, device=hidden.device),
            "log_gap_prev": torch.zeros(seq_items.shape, dtype=hidden.dtype, device=hidden.device),
            "log_gap_next": torch.zeros(seq_items.shape, dtype=hidden.dtype, device=hidden.device),
            "valid_mask": valid_mask,
            "rho_reg_loss": zero_loss,
            "tail_recon_loss": zero_loss,
            "rho_mean": zero_scalar,
            "rho_std": zero_scalar,
            "rho_min": zero_scalar,
            "rho_max": zero_scalar,
            "r_mean": zero_scalar,
            "r_std": zero_scalar,
            "r_min": zero_scalar,
            "r_max": zero_scalar,
            "valid_proto_ratio": zero_scalar if valid_mask.any() else zero_scalar,
            "tail_position_ratio": zero_scalar,
            "timestamps_is_none": torch.as_tensor(1.0, dtype=hidden.dtype, device=hidden.device),
            "time_nonzero_ratio": zero_scalar,
            "log_gap_mean": zero_scalar,
            "log_gap_var": zero_scalar,
        }
