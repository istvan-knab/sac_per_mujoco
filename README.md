# Guided Experience Prioritization for Continous Reinforcement Learning :chart_with_upwards_trend:
## Contributors :busts_in_silhouette:

 ### István Gellért Knáb
 
[![Google Scholar](https://img.shields.io/badge/Scholar-Profile-blue?style=flat&logo=google-scholar)](https://scholar.google.com/citations?user=Qil3Q_wAAAAJ&hl=hu&oi=ao)&emsp;
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-brightgreen?style=flat&logo=researchgate)](https://www.researchgate.net/profile/Istvan-Gellert-Knab?ev=hdr_xprf)&emsp;[![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0009-0007-6906-3308)
### Bálint Kővári

[![Google Scholar](https://img.shields.io/badge/Scholar-Profile-blue?style=flat&logo=google-scholar)](https://scholar.google.com/citations?user=WrtttXEAAAAJ&hl=hu&oi=ao)&emsp;
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-brightgreen?style=flat&logo=researchgate)](https://www.researchgate.net/profile/Balint-Kovari-3)&emsp;[![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-2178-2921)
## Short description :grey_question:

The computational duration required for training contemporary deep learning architectures frequently extends to multiple days of processing time. Given the correlation between temporal requirements, energy consumption, and their associated financial expenditures, the optimization and reduction of these resource-intensive parameters represents a critical challenge in the field of deep learning research.
In the case of reinforcement learning, this includes unique methods such as sample prioritization, where the agent selects from the training samples generated during the process, focusing on those most worth incorporating to accelerate and stabilize convergence.
Previous research has already proven that there are more effective solutions than uniform sampling, as not all samples carry the same amount of information. During sample prioritization, it is equally important to use the best samples while ensuring exploration, with the balance between the two largely depending on the difficulty of the environment and the nature of the action space. The following research presents a solution for continuous action space reinforcement learning agents, where both exploration and exploitation are integral components of the prioritization metric design. This approach accelerates convergence while mitigating unsuccessful training caused by getting stuck in initial local minima.

## Environments :deciduous_tree:
### Pendulum-v1
<img align="right" width="300" height="300" src="https://gymnasium.farama.org/_images/pendulum.gif">

The Pendulum environment is a classic control problem that involves learning to swing up and balance an underactuated pendulum. Unlike the standard pendulum problem where the goal is to simply keep the pendulum upright, this version requires the agent to learn both the swing-up and balancing behaviors.
The system consists of a pendulum attached to a fixed point, with angular position θ and angular velocity ω. The agent can apply torque τ to the pendulum's pivot point. 

__Action Space:__ The action space is continuous, represented by a single value:

&emsp;τ ∈ [-2.0, 2.0] (applied torque)

__The observation space:__ consists of 3 continuous values:&emsp;&emsp;&emsp;__Reward:__

&emsp; cos(θ) ∈ [-1, 1]&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;r= -(θ² + 0.1×θ̇² + 0.001×τ²)

&emsp; sin(θ) ∈ [-1, 1]

&emsp; θ̇ ∈ [-8, 8] (angular velocity)

### InvertedPendulum-v5
<img align="right" width="300" height="300" src="https://gymnasium.farama.org/_images/inverted_pendulum.gif">

A pole is attached by an unactuated joint to a cart, which moves along a frictionless track. The goal is to balance the pole upright by applying forces to the cart. Unlike Pendulum-v1, this system is underactuated with the force applied to the cart rather than directly to the pendulum joint.

__Action Space:__ Continuous, represented by a single value:

&emsp;F ∈ [-3.0, 3.0] (force applied to cart)

__Observation Space:__ 4 continuous values:

&emsp;x ∈ [-4.8, 4.8] (cart position)

&emsp;θ ∈ [-0.418, 0.418] (pole angle)

&emsp;ẋ ∈ [-∞, ∞] (cart velocity)

&emsp;θ̇ ∈ [-∞, ∞] (pole angular velocity)

__Reward Function:__

&emsp;r = 1.0 for every timestep the pole remains upright


### InvertedDoublePendulum-v5
<img align="right" width="300" height="300" src="https://gymnasium.farama.org/_images/inverted_double_pendulum.gif">

A double pendulum attached to a cart moving on a track. The goal is to balance both pendulums upright by applying forces to the cart. This creates a more complex control problem than the single inverted pendulum due to nonlinear dynamics.

__Action Space:__ Continuous, single value:

&emsp;F ∈ [-1.0, 1.0] (force applied to cart)

__Observation Space:__ 11 continuous values:

&emsp;x (cart position)

&emsp;sin(θ₁), cos(θ₁) (first pendulum angle)

&emsp;sin(θ₂), cos(θ₂) (second pendulum angle)

&emsp;ẋ (cart velocity)

&emsp;θ̇₁ (first pendulum angular velocity)

&emsp;θ̇₂ (second pendulum angular velocity)

&emsp;Force applied to cart

&emsp;Joint reaction forces

__Reward Function:__

&emsp;r = height_penalty + angle_penalty + velocity_penalty


### HalfCheetah-v5
<img align="right" width="300" height="300" src="https://github.com/istvan-knab/sac_per_mujoco/blob/main/models/pictures/half_cheetah.gif">
A 2D cheetah simulation with six actuated joints. The goal is to make the cheetah run forward as fast as possible. The system comprises a planar biped robot with torque control at each joint.

__Action Space:__ Continuous, 6 values:

&emsp;τ ∈ [-1, 1]⁶ (joint torques)

&emsp;Control each joint: [hip, thigh, knee] × 2 legs

__Observation Space:__ 17 continuous values:

&emsp;Joint angles (6)

&emsp;Joint velocities (6)

&emsp;Root position (2)

&emsp;Root velocity (2)

&emsp;Height above ground (1)

__Reward Function:__
&emsp;r = forward_reward - control_cost


&emsp;forward_reward = velocity_x
&emsp;control_cost = 0.1 × ||actions||²


<br>


<img align="left" width="233" height="54" src="https://n120.njszt.hu/img/logo/HUN-REN-SZTAKI-logo.png"><img align="center" width="320" height="54" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTOxDZ7qR1tAq3oLAfpg6bB2lL_hAyUwIwWQ&s"><img align="right" width="233" height="54" src="https://www.bme.hu/sites/default/files/mediakit/bme_logo_nagy.jpg">



