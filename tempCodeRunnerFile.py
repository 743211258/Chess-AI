        for state, (pi_indices, pi_probs), z in training_data:
            # 先转为 list，然后再确保其中每个元素是 float
            if isinstance(pi_indices, torch.Tensor):
                pi_indices = pi_indices.tolist()
            if isinstance(pi_probs, torch.Tensor):
                pi_probs = pi_probs.tolist()
            else:
                # 如果是 list of Tensor，也需要处理
                pi_probs = [p.item() if isinstance(p, torch.Tensor) else float(p) for p in pi_probs]

            serializable_data.append({
                'state_fen': state,
                'pi_indices': list(pi_indices),  # 整数索引
                'pi_probs': pi_probs,            # float 概率
                'z': int(z)                      # 保证 z 是 int
            })