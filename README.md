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

Description
The Pendulum environment is a classic control problem that involves learning to swing up and balance an underactuated pendulum. Unlike the standard pendulum problem where the goal is to simply keep the pendulum upright, this version requires the agent to learn both the swing-up and balancing behaviors.
The system consists of a pendulum attached to a fixed point, with angular position θ and angular velocity ω. The agent can apply torque τ to the pendulum's pivot point. Due to gravity and the applied torque, the pendulum moves according to the following dynamics:
System Dynamics
The equations of motion for the pendulum are:
- ml²θ̈ + bθ̇ + mgl sin(θ) = τ

__Action Space:__ The action space is continuous, represented by a single value:
- τ ∈ [-2.0, 2.0] (applied torque)

__The observation space:__ consists of 3 continuous values:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__Reward Function:__
- cos(θ) ∈ [-1, 1]&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;r= -(θ² + 0.1×θ̇² + 0.001×τ²)
- sin(θ) ∈ [-1, 1]
- θ̇ ∈ [-8, 8] (angular velocity)

### InvertedPendulum-v5
<img align="right" width="300" height="300" src="https://gymnasium.farama.org/_images/inverted_pendulum.gif">
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

### InvertedDoublePendulum-v5
<img align="right" width="300" height="300" src="https://gymnasium.farama.org/_images/inverted_double_pendulum.gif">
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

### HalfCheetah-v5
<img align="right" width="300" height="300" src="https://github.com/istvan-knab/sac_per_mujoco/blob/main/models/pictures/half_cheetah.gif">
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.



<img align="left" width="369" height="85" src="https://n120.njszt.hu/img/logo/HUN-REN-SZTAKI-logo.png">

<img align="right" width="369" height="100" src="https://www.bme.hu/sites/default/files/mediakit/bme_logo_nagy.jpg">

