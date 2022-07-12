from collections import deque
from typing import Iterable, List
import numpy as np


class Embedding:

    def __init__(self, shape, main_attention=0.8) -> None:
        assert len(shape) == 1
        self._vec = np.random.random(size=shape) * (1 - main_attention)
        main_idx = np.random.randint(0, shape[0], 1)
        self._vec[main_idx] = main_attention
        self._norm()
        
    def _norm(self) -> None:
        return np.linalg.norm(self._vec)
    
    @property
    def topics(self) -> np.ndarray:
        return self._vec


class Item(Embedding):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self._price = np.random.choice([6, 30, 128, 328, 648])


class User(Embedding):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self._browse = []
        self._history = deque(maxlen=50)
        self._linear_alpha = 0.07
        self._linear_beta = 0.8
        self._quadratic_mu = np.random.random()
        self._quadratic_sigma = np.random.random()
        self._linear_ratio = 1.0 # np.random.random()
        self._stat = {}
    
    def _diversity(self):
        length = len(self._browse)
        if length <= 1:
            return 0
        ent = 0
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                ent += np.sum(self._browse[i].topics * np.log(self._browse[i].topics / self._browse[j].topics + 1e-9))
        return ent / length / (length - 1)
    
    def _update(self):
        return
    
    def _get_click_rate(self, item: Item):
        assert isinstance(item, Item)
        ctr = np.sum(self.topics * item.topics)
        return ctr * 0.5
    
    def _get_stay_rate(self):
        diversity = self._diversity()
        linear_stay_prob = self._linear_alpha * diversity + self._linear_beta
        quadratic_stay_prob = np.exp(-(diversity - self._quadratic_mu) ** 2 / (2 * self._quadratic_sigma ** 2 + 1e-9))
        return linear_stay_prob * self._linear_ratio + quadratic_stay_prob * (1 - self._linear_ratio), diversity
    
    @property
    def history(self):
        return self._history
    
    def expose(self, item: Item):
        self._browse.append(item)
        self._history.append(item)
        click_rate = self._get_click_rate(item)
        stay_rate, diversity = self._get_stay_rate()
        click = np.random.random() < click_rate
        stay = np.random.random() < stay_rate
        self._update()
        self._stat = {
            "click_rate": click_rate,
            "stay_rate": stay_rate,
            "diversity": diversity,
        }
        return click, stay, click_rate, stay_rate, diversity
    
    @property
    def stat(self):
        return self._stat


class RecoEnv:

    def __init__(self, user: User, items: List[Item]):
        self._user = user
        self._items = items
        self._done = False
        self._env_step = 0
    
    def step(self, idx=None):
        self._env_step += 1
        if idx:
            assert 0 <= idx < len(self._items), "Illegal step: wrong item idx!"
        else:
            idx = np.random.randint(0, len(self._items))
        reward, stay, click_rate, stay_rate, diversity = self._user.expose(self._items[idx])
        if not stay or self._env_step >= 100:
            self._done = True
        extra_info = {}
        extra_info["click_rate"] = click_rate
        extra_info["stay_rate"] = stay_rate
        extra_info["diversity"] = diversity
        return idx, reward, extra_info
    
    def total_step(self):
        return self._env_step
    
    @property
    def done(self):
        return self._done


def stat(data):
    return np.mean(data), np.std(data), np.min(data), np.max(data)

def main(collect_cnt):
    items = [Item((10, )) for _ in range(100)]
    users = [User((10, )) for _ in range(100)]
    data = []
    total_step = []
    while True:
        for uid in range(len(users)):
            user = users[uid]
            env = RecoEnv(user, items)
            while not env.done:
                tid, reward, extra_info = env.step()
                data.append([uid, tid, reward, extra_info])
            total_step.append(env.total_step())
            if len(data) > collect_cnt:
                return data, total_step


if __name__ == "__main__":
    collect_cnt = 1000
    data, total_step = main(collect_cnt)
    click_cnt = 0
    avg_ctr = []
    avg_stay_rate = []
    avg_diversity = []
    for d in data:
        if d[2]:
            click_cnt += 1
        avg_ctr.append(d[3]["click_rate"])
        avg_stay_rate.append(d[3]["stay_rate"])
        avg_diversity.append(d[3]["diversity"])
    print("total records {}, click count {}".format(len(data), click_cnt))
    print("user total steps", total_step)
    print("avg_ctr: avg {:.4f}, std {:.4f}, min {:.4f} max {:.4f}".format(*stat(avg_ctr)))
    print("avg_stay_rate: avg {:.4f}, std {:.4f}, min {:.4f} max {:.4f}".format(*stat(avg_stay_rate)))
    print("avg_diversity: avg {:.4f}, std {:.4f}, min {:.4f} max {:.4f}".format(*stat(avg_diversity)))




