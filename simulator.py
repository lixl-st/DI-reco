from collections import defaultdict, deque
from dataclasses import dataclass
import os
import pickle
from typing import Dict
from unicodedata import category
import numpy as np


@dataclass
class UserRecord:
    item_id: int
    action: str

class Embedding:

    def __init__(self, shape, main_attention=0.8) -> None:
        assert isinstance(shape, int)
        self._vec = np.random.random(shape) * (1 - main_attention)
        self._main_topics = np.random.randint(0, shape)
        self._vec[self._main_topics] = main_attention
        self._norm()
        
    def _norm(self) -> None:
        return np.linalg.norm(self._vec)
    
    @property
    def topics(self) -> np.ndarray:
        return self._vec


class Item(Embedding):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self._price = np.random.choice([6, 30, 128, 328, 648], p=[0.5, 0.25, 0.13, 0.07, 0.05])
    
    @property
    def price(self) -> float:
        return self._price
    
    @property
    def category(self) -> int:
        return self._main_topics * 10 + np.random.randint(0, 10)


class User(Embedding):
    def __init__(self, shape) -> None:
        super().__init__(shape)
        self._history = deque(maxlen=50)
        self._return_days = np.random.randint(0, 5)
        self._stay_rate = 0.85 + np.random.random() * 0.11
        self._base_budget = np.exp(np.abs(np.random.randn()) * 2.0 + 3.0)
        self._need_reset = True
        self.reset()
    
    @property
    def history(self):
        return self._history
    
    @property
    def browse_depth(self):
        return len(self._browse)
    
    def reset(self):
        if not self._need_reset:
            return
        self._browse = []
        self._stat = {
            "pv": 0,
            "click": 0,
            "buy": 0,
            "buy_rate": [],
            "click_rate": [],
            "stay_rate": [],
        }
        self._stay_rate = np.clip(self._stay_rate, 0.85, 0.96)
        self._budget = (self._base_budget * np.exp(np.random.randn() * 0.15)).astype(int)
        self._need_reset = False
    
    def request(self):
        if self._return_days > 0:
            self._return_days -= 1
            return False
        return True
    
    def _update_stay_rate(self, click, buy):
        val = -0.02 * np.random.random()
        if buy:
            val += 0.12
        elif click:
            val += 0.04
        self._stay_rate = np.clip(self._stay_rate + val, 0.8, 0.98)
    
    # def _diversity(self):
    #     length = len(self._browse)
    #     if length <= 1:
    #         return 0
    #     ent = 0
    #     for i in range(length):
    #         for j in range(length):
    #             if i == j:
    #                 continue
    #             ent += np.sum(self._browse[i].topics * np.log(self._browse[i].topics / self._browse[j].topics + 1e-9))
    #     return ent / length / (length - 1)
    
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
    
    def _update_stat(self, action: str, extra_info: Dict):
        for key, val in extra_info.items():
            self._stat[key].append(val)
        self._stat[action] += 1
    
    def _get_full_stat(self):
        if self._return_days <= 0:
            return None
        stat = {}
        for key, val in self._stat.items():
            if isinstance(val, list):
                stat[key] = np.mean(val)
            else:
                stat[key] = val
        return stat
    
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
        self._update_stat(action, extra_info)
        if not stay:
            self._return_days = np.random.randint(0, 5)
            self._need_reset = True
            # print(self._get_full_stat())
        return action, stay, extra_info


class RecoEnv:

    def __init__(self, dir=None):
        if not dir:
            self._users = [User(8) for _ in range(10000)]
            self._items = [Item(8) for _ in range(1000)]
        else:
            with open(os.path.join(dir, "users.pickle"), "rb") as f:
                self._users = pickle.load(f)
            with open(os.path.join(dir, "items.pickle"), "rb") as f:
                self._items = pickle.load(f)
    
    def _step(self, user_id, item_id):
        assert 0 <= user_id < len(self._users), "Illegal user id!"
        assert 0 <= item_id < len(self._items), "Illegal item id!"
        action, stay, extra_info = self._users[user_id].expose(self._items[item_id])
        return action, stay, extra_info
    
    def _rollout(self):
        data = []
        for user_id, user in enumerate(self._users):
            user.reset()
            while user.request():
                item_id = np.random.randint(0, len(self._items))
                action, stay, extra_info = self._step(user_id, item_id)
                category = self._items[item_id].category
                data.append([user_id, item_id, category, action, extra_info])
        return data
    
    def save(self, dir, max_step=20):
        with open(os.path.join(dir, "simu.csv"), "w") as f:
            f.write("user_id,item_id,action\n")
            for _ in range(max_step):
                data = self._rollout()
                data = [",".join(d) for d in data]
                f.writelines(data)
        with open(os.path.join(dir, "users.pickle"), "wb") as f:
            self._users = pickle.load(f)
        with open(os.path.join(dir, "items.pickle"), "wb") as f:
            self._items = pickle.load(f)
    
    def run(self, max_step=20):
        data = []
        for step in range(max_step):
            print("current step:", step)
            data.extend(self._rollout())
        return data


def stat(data):
    return np.mean(data), np.std(data), np.min(data), np.max(data)

def main():
    reco = RecoEnv()
    data = reco.save("simulation")
    print(len(data))

if __name__ == "__main__":
    main()




