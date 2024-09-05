###===========================================================================================
###===============================environment.py==============================================
###===========================================================================================
### This file contains the environment class that is used in Reinforcement Learning.
### The environment class is used to define the environment in which the agent will
### interact with.
###===========================================================================================


#=============================================================================================
#=================================Importing Libraries=========================================
#=============================================================================================
import numpy as np
import gymnasium
import gymnasium.spaces
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
#=============================================================================================


#=============================================================================================
#=================================Global Variables============================================
#=============================================================================================
NEGATIVE_REWARD = -10**6
MINIMUM_INVESTMENT = 0.1
#=============================================================================================


#=============================================================================================
#==================================Environment Class==========================================
#=============================================================================================
class Environment(gymnasium.Env):
    """
    The Environment class is used to define the environment in which the agent will interact with.
    
    Attributes:
        data (np.ndarray): The data of the environment.
        sigma (float): The risk trade-off parameter. Default: 0.1.
        window_size (int > 1): The size of the window. If window_size = -1, the window size will grow. Else, the window size will be fixed. Default: -1.
        initialize (int): The seed used for numpy random. Default: -1.
        random_nn (bool): The boolean for randomizing the neural network. Default: False.
        random_data (bool): The boolean for randomizing the data. Default: False.
        cardinality_constraint_mode (str): The cardinality constraint. Default: None.
        cardinality_constraint (int): The cardinality constraint. Default: None.
        shrinkage (bool): The boolean for using the shrinkage method. Default: True.
        lambda_regularization (int): The lambda regularization parameter. Default: 0.
        current_step (int): The initial step of the environment. Default: 1.

        share_dim (int): The number of shares, which is the number of columns of the data.
        max_step (int): The maximum step of the environment, which is the number of rows of the data.
        action_space (gymnasium.spaces.Box): The action space. Represents the new portfolio weights.
        current_step (int): The current step of the environment.
        memory_expected_return (list): The memory of the expected return.
        memory_weights (list): The memory of the weights.
        memory_variance (list): The memory of the covariance.
        observation_space (gymnasium.spaces.Box): The observation space. Contains the reward, expected return, the covariance and the current weights.
        state (np.ndarray): The current state of the environment.
        terminate (bool): The termination status of the environment.
        reward (float): The reward of the environment.  
    """

    def __init__(self, data: np.ndarray, sigma: float = 0.1, window_size: int = -1, random_nn: bool = False, random_data: bool= False, cardinality_constraint_mode: str = None, cardinality_constraint: int = None, shrinkage: bool = True, lambda_regularization: int = 0, current_step: int = 1) -> None:
        super().__init__()
        self.data = data
        self.sigma = sigma
        self.window_size = window_size
        self.random_nn = random_nn
        self.random_data = random_data
        self.cardinality_constraint_mode = cardinality_constraint_mode
        self.cardinality_constraint = cardinality_constraint
        self.shrinkage = shrinkage
        self.lambda_regularization = lambda_regularization
        self.current_step = current_step

        self.share_dim = self.data.shape[1]
        self.max_step = self.data.shape[0]
        self.current_step = 1
        self.action_space = gymnasium.spaces.Box(low = 0.0, high = 100.0, shape = (self.share_dim, ), dtype = np.float32)
        self.observation_space = gymnasium.spaces.Box(low = -np.inf, high = np.inf, shape = (2 * self.share_dim + self.share_dim ** 2 + 1, ), dtype = np.float32)
        self.terminate = False
        self.reward = 0.0
        self.memory_expected_return = []
        self.memory_variance = []
        self.memory_weights = []

        weights = np.random.rand(self.share_dim)
        weights /= np.sum(weights)
        mu, sigma = self.get_mu_sigma()
        reward = self.sigma * mu @ weights - (1 - self.sigma) * (weights @ sigma @ weights) - self.lambda_regularization * np.linalg.norm(weights, 2) ** 2
        self.state = np.concatenate(([reward], mu, sigma.values.flatten(), weights))
        if self.window_size == self.data.shape[0]:
            self.mu, self.Sigma = self.get_mu_sigma()

    
    def get_mu_sigma(self) -> tuple:
        """
        Get the (shrinked) expected return and the (shrinked) covariance of the environment given the data and the window_size if provided.
        Shrinkage method is used to estimate the expected return and the covariance and reduce error and noise.
        Shrinkage method is based on the work of Jorion (1986).
        
        Returns:
            np.ndarray: The expected return.
            np.ndarray: The covariance.
        """
        if self.window_size == -1:
            subdata = self.data.iloc[:self.current_step + 1, :]
        elif self.window_size == self.data.shape[0]:
            subdata = self.data
        else:
            subdata = self.data.iloc[max(0, self.current_step - self.window_size + 1):self.current_step + 1, :]
        mu = subdata.mean()
        sigma = subdata.cov()
        if self.shrinkage:
            m = len(subdata)
            n = self.share_dim
            try:
                sigma_s = (m - 1) / (m - n - 2) * sigma
                z = la.solve(sigma_s, np.ones(n), assume_a='pos')
                mu_0 = mu @ z / z.sum()
                d = mu - mu_0 * np.ones(n)
                y = la.solve(sigma_s, d, assume_a='pos')
                w = (n + 2) / (n + 2 + m * d @ y)
                mu_s =  (1 - w) * mu + w * mu_0 * np.ones(n)
            except:
                mu_s = mu
                sigma_s = sigma
            return mu_s, sigma_s
        else:
            return mu, sigma
    

    def reset(self, seed = None) -> np.ndarray:
        """
        Reset the environment to the initial state.
        
        Args:
            seed (int): The seed to reset the environment. Default: None.
        
        Returns:
            np.ndarray: The initial state of the environment.
            dict: The information of the environment. Empty.
        """
        self.current_step = 1
        self.reward = 0.0
        self.terminate = False
        self.memory_expected_return = []
        self.memory_variance = []
        self.memory_weights = []
        weights = np.random.rand(self.share_dim)
        weights /= np.sum(weights)
        mu, sigma = self.get_mu_sigma()
        reward = self.sigma * mu @ weights - (1 - self.sigma) * (weights @ sigma @ weights) - self.lambda_regularization * np.linalg.norm(weights, 2) ** 2
        self.state = np.concatenate(([reward], mu, sigma.values.flatten(), weights))
        return self.state, {}

    
    def randomize_nn(self) -> None:
        """
        Randomize the neural network: only the position of a share in the state is randomly permuted (in expected return, covariance and weights at the same time)
        """
        if self.random_nn:
            random_index = np.random.permutation(self.share_dim)
            tmp_state = self.state[1:]
            new_state = tmp_state.reshape(-1, self.share_dim)
            new_state = new_state[:, random_index].flatten()
            self.state = np.concatenate(([self.state[0]], new_state))
        return None


    def randomize_data(self) -> None:
        """
        Randomize the data: only the rows of the data are randomly permuted
        """
        if self.random_data:
            random_index = np.random.permutation(self.data.shape[0])
            self.data = self.data.iloc[random_index, :]
        return None


    def apply_cardinality_constraint(self, action: np.ndarray, num_nonzero_actions: int, total_action_sum: float, mu, sigma) -> np.ndarray:
        """
        Apply the cardinality constraint to the action.
        
        Args:
            action (np.ndarray): The action to apply the cardinality constraint.
            total_action_sum (float): The sum of the action.
            mu (pd.Series): The expected return.
            sigma (pd.Dataframe): The covariance matrix.
        
        Returns:
            np.ndarray: The new action with the cardinality constraint.
            dict: The information computed during the cardinality constraint, 
        """
        info = {}

        if self.cardinality_constraint_mode == "renormalize" and num_nonzero_actions > self.cardinality_constraint:
            sorted_index = np.argsort(action)[-self.cardinality_constraint:]
            weights = np.zeros(self.share_dim)
            weights[sorted_index] = action[sorted_index]
            weights /= np.sum(weights)
        else:
            weights = action / total_action_sum

        if self.cardinality_constraint_mode == "contribution" and num_nonzero_actions > self.cardinality_constraint:
            contributions = self.get_contributions(mu, sigma, action)
            sorted_index = np.argsort(contributions)[-self.cardinality_constraint:]
            weights = np.zeros(self.share_dim)
            weights[sorted_index] = action[sorted_index]
            total_weight_sum = np.sum(weights)
            if total_weight_sum < 0.0001:
                non_zero_contributions = contributions[contributions != 0]
                second_largest_value = np.sort(non_zero_contributions)[-1]
                second_largest_index = np.where(contributions == second_largest_value)[0][0]
                weights[second_largest_index] = action[second_largest_index]
                total_weight_sum = np.sum(weights)
            weights /= total_weight_sum

        # if self.cardinality_constraint_mode == "orthogonal_bandit":
        #     components, diag = self.decomposition_PCA(sigma)
        #     reward_components = components.T @ mu.values / np.sqrt(diag)
        #     arm = np.argmax(reward_components)
        #     weights = np.zeros(self.share_dim)
        #     top_contributions = np.argsort(components[arm])[-self.cardinality_constraint:]
        #     weights[top_contributions] = components[arm][top_contributions]
        #     weights /= np.sum(weights)
           
        if self.cardinality_constraint_mode == "kernel_search":
            contributions = self.get_contributions(mu, sigma, action)
            index_list = np.argsort(contributions)[-self.cardinality_constraint//2:]
            pool = np.arange(self.share_dim)
            pool = np.delete(pool, index_list)
            while index_list.shape[0] < self.cardinality_constraint:
                np.random.shuffle(pool)
                basket_size = min(self.cardinality_constraint //4, self.cardinality_constraint - index_list.shape[0])
                basket = [pool[i:i + basket_size] for i in range(0, len(pool), basket_size)]
                best_basket = None
                best_reward = -np.inf
                for b in basket:
                    new_index_list = np.concatenate([index_list, b])
                    new_weights = np.zeros(self.share_dim)
                    new_weights[new_index_list] = action[new_index_list]
                    new_weights /= np.sum(new_weights)
                    new_return = mu @ new_weights
                    new_risk = new_weights @ sigma @ new_weights
                    new_regularization = self.lambda_regularization * np.linalg.norm(new_weights, 2) ** 2
                    new_reward = self.sigma * new_return - (1 - self.sigma) * new_risk - new_regularization
                    if new_reward > best_reward:
                        best_reward = new_reward
                        best_return = new_return
                        best_risk = new_risk
                        best_basket = b
                index_list = np.concatenate([index_list, best_basket])
                pool = np.setdiff1d(pool, best_basket)
            weights = np.zeros(self.share_dim)
            weights[index_list] = action[index_list]
            weights /= np.sum(weights)
            info["best_return"] = best_return
            info["best_risk"] = best_risk
            info["best_reward"] = best_reward

        return weights, info




    def step(self, action: np.ndarray) -> tuple:
        """
        Take a step in the environment given an action.
        
        Args:
            action (np.ndarray): The action to take in the environment.
        
        Returns:
            np.ndarray: The new state of the environment.
            float: The reward of the environment.
            bool: The termination status of the environment.
            bool: The troncation status of the environment. False.
            dict: The information of the environment. Empty.
        """
        self.terminate = self.current_step > self.max_step
        if self.terminate:
            return self.state, self.reward, self.terminate, False, {}

        action = (action > MINIMUM_INVESTMENT) * action         # Add minimum investment constraint
        total_action_sum = np.sum(action)
        
        self.randomize_data()
        if self.window_size != self.data.shape[0]:
            mu, sigma = self.get_mu_sigma()
        else:
            mu, sigma = self.mu, self.Sigma

        if total_action_sum < 0.01:
            weights = np.zeros(self.share_dim)
            self.reward = NEGATIVE_REWARD
            self.state = np.concatenate(([self.reward], mu, sigma.values.flatten(), weights))
            self.randomize_nn()
            self.memory_expected_return.append(0.0)
            self.memory_variance.append(0.0)
            self.memory_weights.append(weights)
            self.current_step += 1
            return self.state, self.reward, self.terminate, False, {}
        
        num_nonzero_actions = np.sum(action > 0)
        weights, info = self.apply_cardinality_constraint(action, num_nonzero_actions, total_action_sum, mu, sigma)
        

        if self.cardinality_constraint_mode == "kernel_search":     # avoid recalculation
            expected_return = info["best_return"]
            variance = info["best_risk"]
            self.reward = info["best_reward"]
        else:
            expected_return = mu @ weights
            variance = weights @ sigma @ weights

            regularization = self.lambda_regularization * np.linalg.norm(weights, 2) ** 2

            if self.cardinality_constraint_mode == "negative_reward" and num_nonzero_actions > self.cardinality_constraint:     # Fixed value
                self.reward = NEGATIVE_REWARD
            else:
                self.reward = (self.sigma) * expected_return - (1 - self.sigma) * (variance) - regularization
                if self.cardinality_constraint_mode == "penalty" and num_nonzero_actions > self.cardinality_constraint:         # Linear penalty with the step
                    self.reward += NEGATIVE_REWARD * (self.current_step / self.max_step)

        self.state = np.concatenate(([self.reward], mu, sigma.values.flatten(), weights))
        self.randomize_nn()
        self.memory_expected_return.append(expected_return)
        self.memory_variance.append(variance)
        self.memory_weights.append(weights)
        self.current_step += 1

        return self.state, self.reward, self.terminate, False, {}
    

    def get_memory_expected_return(self) -> list:
        """
        Get the memory of the expected return.
        
        Returns:
            list: The memory of the expected return.
        """
        return self.memory_expected_return
    

    def get_memory_variance(self) -> list:
        """
        Get the memory of the variance.
        
        Returns:
            list: The memory of the variance.
        """
        return self.memory_variance
    

    def get_memory_weights(self) -> list:
        """
        Get the memory of the weights.
        
        Returns:
            list: The memory of the weights.
        """
        return self.memory_weights
    

    def get_state(self) -> np.ndarray:
        """
        Get the state of the environment.
        
        Returns:
            np.ndarray: The state of the environment.
        """
        return self.state
    

    def get_contributions(self, mu, Sigma, weights) -> np.ndarray:
        """
        Get the contributions of the weights.
        
        Args:
            mu (pd.Series): The expected return.
            Sigma (pd.DataFrame): The covariance matrix.
            weights (np.ndarray): The weights.
        
        Returns:
            np.ndarray: The contributions of the weights.
        """
        term1 = self.sigma * mu.values * weights
        term2 = (1 - self.sigma) * (Sigma.values @ weights) * weights
        term3 = self.lambda_regularization * weights
        contributions = term1 - term2 - term3
        contributions[weights == 0] = -np.inf   # Avoiding using shares with 0 weight
        return contributions
    

    def decomposition_PCA(self, Sigma) -> np.ndarray:
        """
        Perform the PCA decomposition of the environment. Will scale the data.
        
        Returns:
            components (np.ndarray): The components of the PCA decomposition.
            diag (np.ndarray): The diagonal of the PCA decomposition, the eigenvalues vector.
        """
        pca = PCA(n_components = self.share_dim)
        pca.fit(Sigma)
        components = normalize(pca.components_.T, axis=0)
        diag = pca.explained_variance_

        return components, diag
    

