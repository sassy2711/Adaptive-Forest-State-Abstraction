import numpy as np
import math
import random
SEED = 101
np.random.seed(SEED)
random.seed(SEED)
class TreeNode:
    def __init__(self, actions, parent=None):
        self.parent = parent
        self.actions = actions

        # MCTS / Q-learning stats
        self.Q    = {a: 0.0 for a in actions}
        self.N_sa = {a: 0   for a in actions}
        self.N_s  = 0

        # for splitting
        self.error_records = []   # list of (state_vector, delta)
        self.min_bound = None     # np.array of per-dimension minima
        self.max_bound = None     # np.array of per-dimension maxima

        # split info (only for internal nodes)
        self.split_dim    = None
        self.split_thresh = None
        self.children     = None  # (left, right)
        self.dormancy     = 0      # number of steps since last action selection
        self.parent=parent



    @property
    def is_leaf(self):
        return self.children is None

    def update_bounds(self, x): # for finding the range of each dimension in input vectors in the node
        x = np.array(x)
        if self.min_bound is None:
            self.min_bound = x.copy()
            self.max_bound = x.copy()
        else:
            self.min_bound = np.minimum(self.min_bound, x)
            self.max_bound = np.maximum(self.max_bound, x)

    def get_range(self):
        # safe even if min_bound == max_bound initially
        return self.max_bound - self.min_bound

    def record_transition(self, s, a,delta):
        self.error_records.append((np.array(s),a, delta))

    def select_action(self, c, epsilon,inference=False):
        if inference:
            return max(self.Q, key=self.Q.get) # select action with max Q value
        best, best_val = None, -1e20
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        for a in self.actions:
            if self.N_s == 0:
                val=self.Q[a] # unvisited state
            if self.N_sa[a] == 0:
                val = self.Q[a]+1e20  # unvisited action
            else:
                val = (self.Q[a] +
                       c * math.sqrt(math.log(self.N_s)/self.N_sa[a]))  # UCT exploration
            if val > best_val:
                best, best_val = a, val
        return best

    def try_split(self, l_th, th_mu, th_sigma):

        if len(self.error_records) < l_th:
            return False
        deltas = np.array([d for _,_, d in self.error_records])
        mean, var = deltas.mean(), deltas.var()

        if abs(mean) > th_mu or var < th_sigma:
            return False


        X = np.stack([s for s,_, _ in self.error_records])
        pos = deltas > 0 # group 1 boolean
        neg = deltas < 0 # group 2 boolean

        # compute modified t-statistic per dimension
        ranges = self.get_range()
        t_scores = np.zeros_like(ranges)
        for i in range(len(ranges)):
            if pos.sum() > 0 and neg.sum() > 0: # if there are elements in both groups
                x1 = X[pos, i]; x2 = X[neg, i]
                m1, m2 = x1.mean(), x2.mean()
                v1, v2 = x1.var()+1e-8, x2.var()+1e-8
                n1, n2 = len(x1), len(x2)


                t_scores[i] = abs(m1 - m2) / math.sqrt(v1/n1 + v2/n2) * ranges[i] # t score to check which dimension to split
            else:
                # fallback: split on largest span
                t_scores[i] = ranges[i]

        dim = int(np.argmax(t_scores)) # the dimension to be split at
        # find threshold between the two group means (Equation 9)
        if pos.sum() > 0 and neg.sum() > 0:
            x1 = X[pos, dim] #group 1 elements along dim
            x2 = X[neg, dim] # group 2 elements along dim
            m1, m2 = x1.mean(), x2.mean()
            s1, s2 = x1.var()+1e-8, x2.var()+1e-8
            p1, p2 = len(x1)/len(X), len(x2)/len(X)

            C = (math.log(p2**2/s2) - math.log(p1**2/s1))

            a = 1/s2 - 1/s1
            b = -2*m2/s2 + 2*m1/s1
            c = m2*m2/s2 - m1*m1/s1 - C
            roots = np.roots([a, b, c]) # solving roots of the ax^2+bx+c=0
            # pick the root between m1 and m2
            candidates = [r for r in roots if min(m1, m2) < r < max(m1, m2)]
            thresh = candidates[0] if candidates else 0.5 * (m1 + m2)
        else: #wrote an else condition just in case one of the groups is empty
           thresh=np.median(X[:,dim])

        # build children
        left  = TreeNode(self.actions, parent=self)
        right = TreeNode(self.actions, parent=self)
        self.split_dim, self.split_thresh = dim, thresh
        self.children = (left, right)

        # inherit Q/N stats
        for child in (left, right):
            child.Q    = self.Q.copy()
            child.N_sa = self.N_sa.copy()
            child.N_s  = self.N_s

        # clear for next split
        self.error_records = []
        return True

    def reset_dormancy(self):
        self.dormancy = 0

    def increment_dormancy(self):
        self.dormancy += 1

    def compute_reliability(self, cr):

        # gather all the recorded deltas
        if self.error_records:
            deltas = np.array([d for _,_, d in self.error_records])
            sigma2 = deltas.var()
        else:
            # if no records yet, assume very high uncertainty -> low reliability
            sigma2 = float('inf')

        # logistic shaped mapping
        return 2.0 / (1.0 + math.exp(- (cr * self.N_s) / (sigma2 + 1e-8)))-1

    def compute_activity(self, ca):

        return 2.0 / (1.0 + math.exp(ca * self.dormancy))



class AdaptiveStateAggregator:
    def __init__(self, actions, **kwargs):
        self.root = TreeNode(actions)
        self.c = kwargs.get("c", 2)      # UCB exploration parameter
        self.gamma = kwargs.get("gamma", 0.98)  # discount factor
        self.alpha = kwargs.get("alpha", 0.8)   # learning rate
        self.l_th = kwargs.get("l_th", 5)        # threshold for number of records to consider splitting
        self.th_mu = kwargs.get("th_mu",0.2)     # threshold on mean error
        self.th_sigma = kwargs.get("th_sigma", 0.01)  # threshold on error variance
        self.cr = kwargs.get("cr", 1.0)       # scaling for reliability computation
        self.ca = kwargs.get("ca", 1.0)       # scaling for activity computation
        self.epsilon= kwargs.get("epsilon", 0) # exploration probability
        self.max_epoch_length=kwargs.get("max_epoch_length", 500)
        self.alpha_decay=kwargs.get("alpha_decay",0.001)

    def get_leaf(self, x):
        node = self.root
        while not node.is_leaf:
            d, t = node.split_dim, node.split_thresh
            node = node.children[0] if x[d] <= t else node.children[1]
        node.update_bounds(x)
        return node

    def run_episode(self, env, max_steps=5000):
        state, _ = env.reset()
        epoch = []
        start_leaf = self.get_leaf(state)
        a = start_leaf.select_action(self.c,self.epsilon)
        start_leaf.N_s += 1
        start_leaf.reset_dormancy()
        steps = 0
        done = False

        total_reward = 0
        delta_log = []

        while not done and steps < max_steps and len(epoch) <= self.max_epoch_length:

            next_state, r, terminated, truncated, _ = env.step(a)
            leaf = self.get_leaf(next_state)
            done = terminated or truncated
            epoch.append((state.copy(), a, r))
            total_reward += r
            state = next_state
            steps += 1
            if leaf is not start_leaf or done or steps == max_steps:

                tau = len(epoch)
                if tau == 0:
                    continue


                V_next = 0.0 if done else max(leaf.Q.values())
                deltas = []
                for k, (s_k, a_k, _) in enumerate(epoch):
                    rewards = [epoch[i][2] for i in range(k, tau)]

                    R_k = sum((self.gamma ** i) * r for i, r in enumerate(rewards))
                    assert a_k==a, "Action index mismatch"
                    delta_k = R_k + (self.gamma ** len(rewards)) * V_next - start_leaf.Q[a_k]


                    deltas.append(delta_k)
                    delta_log.append(delta_k)
                    start_leaf.record_transition(s_k,a_k, delta_k)

                if deltas:
                    mean_delta = sum(deltas) / tau
                    start_leaf.N_sa[a] += 1
                    start_leaf.Q[a] += self.alpha * mean_delta
                    start_leaf.try_split(self.l_th, self.th_mu, self.th_sigma)



                epoch = []
                start_leaf = leaf
                a = start_leaf.select_action(self.c,self.epsilon)
                start_leaf.N_s += 1
                start_leaf.reset_dormancy()



        self._increment_all_dormancy(self.root)
        # if count_states(self.root) > 2000:
        #     self.prune(self.root)
        self.max_epoch_length=min(self.max_epoch_length*1.05,1000) # increase max epoch length
        self.epsilon=max(0.01, self.epsilon * 0.98) # decay epsilon
        self.alpha=max(0.01, self.alpha - self.alpha_decay) # decay alpha
        return total_reward, delta_log

    def _increment_all_dormancy(self, node):
        if node.is_leaf:
            node.increment_dormancy()
        else:
            for child in node.children:
                self._increment_all_dormancy(child)


    def prune(self, node):
        if not node.is_leaf:
            left, right = node.children
            self.prune(left)
            self.prune(right)
            if left.is_leaf and right.is_leaf:
                best_l = max(left.Q, key=left.Q.get)
                best_r = max(right.Q, key=right.Q.get)
                if best_l == best_r:
                    rel_l = left.compute_reliability(self.cr)
                    rel_r = right.compute_reliability(self.cr)
                    act_l = left.compute_activity(self.ca)
                    act_r = right.compute_activity(self.ca)
                    if rel_l > act_l and rel_r > act_r:
                        node.children = None
                        node.error_records = []
            else:
                self.prune(left)
                self.prune(right)
    def train(self, env, episodes):
        rewards=[]
        inference_per_10=[]
        for ep in range(episodes):
            reward, deltas = self.run_episode(env)
            rewards.append(reward)
            if(ep+1) % 10 == 0:
                temp=[]
                for i in range(50):
                    temp.append(self.inference(env))
                inference_per_10.append(np.mean(temp))
        return rewards, inference_per_10


    def inference(self, env):

        state, _ = env.reset()
        leaf = self.get_leaf(state)
        cumulative_reward = 0.0
        while True:
            a = leaf.select_action(self.c,0,inference=True)
            next_state, r, terminated, truncated, _ = env.step(a)
            cumulative_reward += r
            leaf = self.get_leaf(next_state)
            if terminated or truncated:
                break

        return cumulative_reward

class AdaptiveForestAgent:
    def __init__(self, n_trees, state_dim, actions, feature_fraction=0.7, **tree_kwargs):
        self.trees = []
        self.feature_subsets = []
        self.actions = actions
        self.state_dim = state_dim

        for _ in range(n_trees):
            k = int(feature_fraction * state_dim)
            feature_indices = np.random.choice(state_dim, k, replace=False)
            self.feature_subsets.append(feature_indices)
            tree = AdaptiveStateAggregator(actions, **tree_kwargs)
            self.trees.append(tree)

    def project_state(self, state, indices):
        return np.array(state)[indices]

    def train(self, env, episodes):
        inference_per_10=[]
        for ep in range(episodes):
            for i, tree in enumerate(self.trees):
                # wrap the env to project state based on this tree's features
                proj_env = ProjectedEnv(env, self.feature_subsets[i])
                reward, deltas = tree.run_episode(proj_env)
            if (ep+1)%10==0:
                temp=[]
                for i in range(50):
                    temp.append(self.inference(env))
                inference_per_10.append(np.mean(temp))

        return inference_per_10

    def inference(self, env,num_trees=None):
        if num_trees is None:
            num_trees = len(self.trees)
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:

            Qs = np.zeros(len(self.actions))
            for tree, indices in zip(self.trees[:num_trees], self.feature_subsets[:num_trees]):
                proj_state = self.project_state(state, indices)
                leaf = tree.get_leaf(proj_state)
                for a in self.actions:
                    Qs[a] += leaf.Q[a]
            Qs /= num_trees
            a = np.argmax(Qs)
            state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += r
        return total_reward
    
class ProjectedEnv:
    def __init__(self, env, indices):
        self.env = env
        self.indices = indices

    def reset(self):
        obs, info = self.env.reset()
        return np.array(obs)[self.indices], info

    def step(self, action):
        next_obs, r, terminated, truncated, info = self.env.step(action)
        return np.array(next_obs)[self.indices], r, terminated, truncated, info

def count_states(node):
    if node.is_leaf:
        return 1
    else:
        return 1 + count_states(node.children[0]) + count_states(node.children[1])