### Policy_gradient:
Policy gradient reinforcement learning (RL) is a class of algorithms used to train agents in sequential decision-making tasks. Unlike value-based methods that aim to learn an optimal value function, policy gradient methods directly optimize the policy of the agent, which is a mapping from states to actions.

In policy gradient RL, the goal is to find a policy that maximizes the expected cumulative reward over time. This is typically done by iteratively improving the policy through gradient ascent on a performance objective. The policy is usually parameterized by a set of learnable parameters, such as neural network weights.

The basic idea behind policy gradient methods is to estimate the gradient of the expected cumulative reward with respect to the policy parameters and update the parameters in a way that increases the expected reward. The gradient is typically estimated using samples collected by interacting with the environment.

The most common approach to estimating the gradient is through the use of the policy gradient theorem. This theorem provides a way to compute the gradient of the expected cumulative reward by taking the gradient of the policy with respect to its parameters and weighting it by an estimate of the return (i.e., the cumulative reward) obtained from a trajectory sampled under the policy.
#### Mathematical Background:
policy graident approach like supervised learning approach tries to optimise the problem based on a objective function J. 
the objective function could be defined as below,

![Screenshot from 2023-10-12 22-30-14](https://github.com/PeymanMawlani1993/Reinforcement-Learning/assets/103693616/1c8376a3-f78c-4819-a983-17f3d6e46e5f)

where γ is the discount factor, r_t is the reward obtained at time step t, and the expectation is taken over the trajectory T.

To update the policy parameters θ, we want to compute the gradient of J(θ). The policy gradient theorem provides a way to compute the gradient using the following equation:

![Screenshot from 2023-10-12 22-34-04](https://github.com/PeymanMawlani1993/Reinforcement-Learning/assets/103693616/f06b462e-791b-48fa-bcbc-8c67d51d3632)

To ensure that exploration is maintained and to ensure that πθ(a| s) does not collapse to a single action with high probability, we introduce a
regularization term known as entropy[1]. Entropy of a distribution is defined as follows:

![Screenshot from 2023-10-12 22-36-07](https://github.com/PeymanMawlani1993/Reinforcement-Learning/assets/103693616/5bd9eed9-1d21-4259-9423-63c2332b8653)

so the final Loss can be defined as (notice the gradient is obtained based on reward to go reinforce method in order to the problem of high variance in PG approach):

![Screenshot from 2023-10-12 22-38-08](https://github.com/PeymanMawlani1993/Reinforcement-Learning/assets/103693616/ea833ed7-e334-4597-b790-af9c0b63a92c)

The update rule for the policy parameters in a basic policy gradient algorithm can be written as:

![Screenshot from 2023-10-12 22-40-09](https://github.com/PeymanMawlani1993/Reinforcement-Learning/assets/103693616/9e72fd35-ce21-478d-a429-f952f94238ee)

where α is the learning rate.
This update rule iteratively improves the policy parameters by ascending the gradient of the expected cumulative reward. By repeatedly sampling trajectories, estimating gradients, and updating the parameters, the policy gradually improves to maximize the expected reward.






