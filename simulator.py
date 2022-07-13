from collections import deque
from dataclasses import dataclass
from typing import List
import pickle
import numpy as np


@dataclass
class UserRecord:
    item: int
    action: str

class Embedding:

    def __init__(self, shape, main_attention=0.8) -> None:
        assert isinstance(shape, int)
        self._vec = np.random.random(shape) * (1 - main_attention)
        main_idx = np.random.randint(0, shape, 1)
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
        self._category = self._vec.reshape(-1, 2).sum(axis=1) + np.random.random()
        self._price = np.random.choice([6, 30, 128, 328, 648], p=[0.5, 0.25, 0.13, 0.07, 0.05])
    
    @property
    def price(self) -> float:
        return self._price


class User(Embedding):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self._history = deque(maxlen=50)
        self._return_days = 0
        self._stay_rate = 0.85 + np.random.random() * 0.11
        self._base_budget = np.exp(np.abs(np.random.randn()) * 2.0 + 3.0)
        self.reset()
    
    def ask(self):
        self._return_days += 1
        return self._return_days >= 0
    
    def reset(self):
        self._browse = []
        self._return_days = 0
        self._stay_rate = np.clip(self._stay_rate, 0.85, 0.96)
        self._budget = (self._base_budget * np.exp(np.random.randn() * 0.15)).astype(int)
    
    def _update_stay_rate(self, click, buy):
        val = -0.02 * np.random.random()
        if buy:
            val += 0.12
        elif click:
            val += 0.04
        self._stay_rate = np.clip(self._stay_rate + val, 0.8, 0.98)
    
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
    
    def _get_click_rate(self, item: Item):
        assert isinstance(item, Item)
        ctr = np.sum(self.topics * item.topics)
        return ctr * 0.66
    
    def _get_buy_rate(self, item: Item):
        if item.price > self._budget:
            return 0.0
        buy_rate = ((self._budget - item.price) / self._budget + 0.1) * np.sum(self.topics * item.topics) ** 2
        return buy_rate
    
    def _get_stay_rate(self):
        return self._stay_rate
    
    @property
    def history(self):
        return self._history
    
    def expose(self, item: Item):
        buy_rate = self._get_buy_rate(item)
        buy = np.random.random() < buy_rate

        click_rate = self._get_click_rate(item)
        click = buy or np.random.random() < click_rate
        
        self._update_stay_rate(click, buy)
        stay_rate = self._get_stay_rate()
        stay = np.random.random() < stay_rate

        action = "pv"
        if buy:
            action = "buy"
        elif click:
            action = "click"

        self._browse.append(UserRecord(item, action))
        self._history.append(UserRecord(item, action))

        extra_info = {
            "buy_rate": buy_rate,
            "click_rate": click_rate,
            "stay_rate": stay_rate,
        }
        return action, stay, extra_info


class RecoEnv:

    def __init__(self, user: User, items: List[Item]):
        self._user = user
        self._items = items
        self._done = False
    
    def step(self, idx=None):
        self._env_step += 1
        if idx:
            assert 0 <= idx < len(self._items), "Illegal step: wrong item idx!"
        else:
            idx = np.random.randint(0, len(self._items))
        user_action, user_stay, extra_info = self._user.expose(self._items[idx])
        if not user_stay:
            self.finish()
        return idx, user_action, extra_info
    
    def total_step(self):
        return self._env_step
    
    def finish(self):
        return self.finish()




def stat(data):
    return np.mean(data), np.std(data), np.min(data), np.max(data)

def main():
    items = [Item() for _ in range(1000)]
    users = [User() for _ in range(10000)]

    with open("./simulation/items", "wb") as f:
        pickle.dump(items, f)
    with open("./simulation/users", "wb") as f:
        pickle.dump(users, f)

    with open("./simulation/simu", "w") as f:
        for _ in range(3):
            for uid in range(len(users)):
                user = users[uid]
                env = RecoEnv(user, items)
                while not env.done:
                    tid, reward, extra_info = env.step()
                    f.writelines("{},{},{}\n".format(uid, tid, reward))
                    uid, tid, reward, extra_info


def main2(collect_count=1000):
    items = [Item() for _ in range(1000)]
    users = [User() for _ in range(10000)]
    data = []

    for uid in range(len(users)):
        user = users[uid]
        env = RecoEnv(user, items)
        while not env.done:
            tid, reward, extra_info = env.step()
            data.append([uid, tid, reward, extra_info])
        if data >= collect_count:
            return 


if __name__ == "__main__":
    main()
    # click_cnt = 0
    # avg_ctr = []
    # avg_stay_rate = []
    # avg_diversity = []
    # for d in data:
    #     if d[2]:
    #         click_cnt += 1
    #     avg_ctr.append(d[3]["click_rate"])
    #     avg_stay_rate.append(d[3]["stay_rate"])
    #     avg_diversity.append(d[3]["diversity"])
    # print("total records {}, click count {}".format(len(data), click_cnt))
    # print("user total steps", total_step)
    # print("avg_ctr: avg {:.4f}, std {:.4f}, min {:.4f} max {:.4f}".format(*stat(avg_ctr)))
    # print("avg_stay_rate: avg {:.4f}, std {:.4f}, min {:.4f} max {:.4f}".format(*stat(avg_stay_rate)))
    # print("avg_diversity: avg {:.4f}, std {:.4f}, min {:.4f} max {:.4f}".format(*stat(avg_diversity)))




