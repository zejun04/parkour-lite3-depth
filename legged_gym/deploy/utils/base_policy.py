import torch


class BasePolicyRunner:
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def act(self, obs_tensor, depth_feature):
        with torch.no_grad():
            return self.base_policy(obs_tensor, depth_feature)


def load_base_policy(base_jit_path, device):
    print(f"Loading base model (JIT) from: {base_jit_path}")
    base_policy = torch.jit.load(base_jit_path, map_location=device).to(device)
    base_policy.eval()
    return BasePolicyRunner(base_policy)
