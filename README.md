[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"
[image3]: https://miro.medium.com/max/644/1*ll9SkIj-JzW1WGvAEp-9wg.png "Q-Table"
[image4]: https://miro.medium.com/max/538/1*raYvY38yWm4E4J3iYqv6dA.png "Bellman Equation"
[image5]: imgs/episodes.png "Model Comparison"
[image6]: imgs/rewards_series.png "Rewards Series"
[image7]: imgs/rolling_rewards_series.png "Rolling Rewards Series"
[image8]: https://paperswithcode.com/media/methods/b6cdb8f5-ea3a-4cca-9331-f951c984d63a_MBK7MUl.png "SARS Memory"
[image9]: https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM.png "Deep Q-Learning"
[image10]: https://images3.programmersought.com/637/5d/5d4d7814d1fcdc7c8c5ffeee151721fd.png "Double Q-Learning"

# Project 1: Navigation

## Learning the Problem

For this project, i trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.<br>
Before we dive in to how i build up the model, let's recap some main concepts right?

## Recap Concepts
Let's follow a sequence so we don't get lost and have a linear knowledge about the literature. Will leave some concepts that i'll talk here, so if there's any doubt about these you guys can search a little more deeper.
- Q-Table
- Exploration / Exploitation Dilemma
- Epsilon Greedy
- Deep Q-Learning
- Experience Replay
- Deep Double Q-Learning

### Q-Table
Basically Q-Table is used to calculate the maximum expected future rewards for action at each state, so the table will lead us to the best action for each state. We will have a table with size equals to number of actions times number of states.<br>
![Q-Table][image3]<br>
Alright so for each value in the Q-Table we call it Q-Value which uses Bellman equation to estimate it.
![Bellman Equation][image4]<br>
The Q-Value is represented by given a state and action we will estimate the sum of the rewards with a discounted factor called gamma which stays between 0 and 1, values closer to 0 will tend to preserve recently rewards than previous rewards.
### Exploration / Exploitation Dilemma
This dilemma is one of the hardest to think about, let's say that each time we play we have to decide between explore even more our enviroment and see which series of actions would lead to a highest rewards or keep what we know about the enviroment and continue to do the action that belongs the highest rewards, now, what should we do? Start exploring our enviroment and more often begins to exploit it? It's there any possible way to estimate when we need to explore the enviroment? Or even a heuristic? Well you will see about Epsilon Greedy in the next section which tries to solve this problem.
### Epsilon Greedy
Now that you know more about exploration / exploitation dilemma we can explain how Epsilon Greedy works, let's say we have a probability for those two actions, what epsilon greedy tries to do it's to generate a randomness into the algorithm, which force the agent to try different actions and not get stuck at a local minimum, so to implemente epsilon greedy we set a epsilon value between 0 and 1 where 0 we never explore but always exploit the knowledge that we already have and 1 do the opposite, after we set a value for epsilon we generate a random value usually from a normal distribution and if that value is bigger than epsilon we will choose the current best action otherwise we will choose a random action to explore the enviroment.
### Deep Q-Learning
Q-Learning is a really good algorithm right? Now let's try to solve bigger problems with many many states and actions, let's suppose we have a problem that consists in 10.000 states and 1.000 actions we will have a table that equals to 10M cell to compute Q-Values, that would be a huge problem to solve computationally speaking and maybe takes too long to get a result, to deal with this problem we have Deep Q-Learning which takes the advantage of Neural Networks to compute Q-Values for each action given the state as the image below.
![Deep Q-Learning][image9]

### Experience Replay
In order to try to solve rare events detection in our model, we store each experience from our agent in a memory and sample it randomly so our agent start to generalize better and recall rare occurrences. Also for better performance we could use mini-batch's to see how our model converge. The image below shows up how figuratively the memory would look.
![SARS Memory][image8]
### Deep Double Q-Learning
One of many issues that we still have in DQN's is they can overestimate our Q-Values but this come with a reasonable idea that since we are using the same sample to determine the max action and the value of this action this creates a bias at our algorithm, to deal with that we use two separated Q-functions to determine the max action and one to determine it's value to eliminate the bias over time. So the implementation in a pseudocode would look like something like the image below:
![Double Q-Learning][image10]
## Solution
My solution was take what i learned from the mini project with LunarLander and try to reproduce it as a generic model but with a little plus that implemented Double Q-Learning so we can see what little changes can make a huge impact on our agent.
Note: Using also 64 neurons for FC layer and Gamma as 0.99 if you wanna go deeper check dqn_agent.py file which is all hyperparameters you need to know.
![Model Comparison][image5]<br>
As far we can see it reduced the number of episodes in our agent but just reduce the number of episodes is cool but are our agent in fact converging faster and what we could do for our next steps.
![Rewards Series][image6]<br>
Ok the rewards series looks kinda messy but let's make it a smoother series so we can see what the agent are learning.
![Rolling Rewards Series][image7]<br>
Well thats good our DDQN agent is learning really faster, maybe because it's on exploration phase, maybe if the goal of max reward was higher the agent could converge faster, but those points let's keep it for the next section.
## Ideas for Future Work
- Do a GridSearch to find optimal hyperparameters for DQN and DDQN models.
- Add Convolutional and Dropout Layers in Neural Network and see if our model finds some patterns at the states.
- Utilize Prioritized Experience Replay and see how much benefit we can extract from a weighted sample.
- Try to implement a Noisy Net so for each output for the neural network that have some noise our agent will explore the enviroment leading to a better performance.


## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
	
2. Since i already created a file with several dependencies that we need to run the project. First of all it's better to install the python folder which contains some stable libraries. To do so follow the next command lines.
```bash
cd python
pip install .
```

3. Now that the env already have some things that we need, let's install the other part of the dependencies
```bash
cd ../
pip install -r requirements.txt
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]