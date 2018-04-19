import numpy as np

from multiagent.core     import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):

        # set any world properties first
        world = World()
        world.dim_c     = 2
        num_agents      = 4
        car_landmarks   = 6
        rew_landmarks   = 4
        num_landmarks   = car_landmarks + rew_landmarks

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name      = 'agent %d' % i
            agent.collide   = True
            agent.silent    = True
            agent.size      = 0.1
            agent.accel     = 3.0
            agent.max_speed = 1

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.car     = True if i < car_landmarks else False
            landmark.size    = 0.1  if landmark.car else 0.02
            landmark.collide = True if landmark.car else False
            landmark.movable = True if landmark.car else False

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):

        posXTok, posYTok = np.meshgrid(np.linspace(-0.8, 0.8, 5), np.linspace(-0.8, 0.8, 5))
        posXTok = np.reshape(posXTok, [1, posXTok.shape[0]*posXTok.shape[1]]).tolist()[0]
        posYTok = np.reshape(posYTok, [1, posYTok.shape[0]*posYTok.shape[1]]).tolist()[0]
        posTok  = list(zip(posXTok, posYTok))
        tgtLane = np.random.choice(10, 1, replace=False)[0]
        tgtPos  = [posTok[tgtLane+(5*i)] for i in range(4)]

        for pos in tgtPos:
            posTok.remove(pos)

        tgtPos  = [np.array(pos) for pos in tgtPos]
        posTok  = [np.array(posTok[i]) for i in np.random.choice(len(posTok),
                                                       len(world.landmarks)+len(world.agents),
                                                       replace=False)]

        # random properties for agents
        for agent in world.agents:
            agent.color = np.array([0.35, 0.85, 0.35])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if landmark.car:
                landmark.color       = np.array([0.85, 0.35, 0.35])
                landmark.state.p_pos = posTok.pop()
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = tgtPos.pop()
            landmark.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = posTok.pop()
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c     = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        rew                = 0
        collisions         = 0
        occupied_landmarks = 0
        min_dists          = 0

        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew       -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew        -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, elem, agent):
        delta_pos = agent.state.p_pos - elem.state.p_pos
        dist      = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min  = agent.size + elem.size

        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in [l for l in world.landmarks if not l.car]:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew  -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            for l in [l for l in world.landmarks if l.car]:
                if self.is_collision(l, agent):
                    rew -= 1

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
