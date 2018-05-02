import numpy as np

from multiagent.core     import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import time

class Scenario(BaseScenario):

    def make_world(self):

        self.viewers = [None]

        # set any world properties first
        world = World()
        world.dim_c     = 2
        num_agents      = 3
        rew_landmarks   = 3
        car_landmarks   = 6
        num_landmarks   = car_landmarks + rew_landmarks

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name      = 'agent %d' % i
            agent.collide   = True
            agent.silent    = True
            agent.size      = 0.1
            agent.accel     = 2
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
        tgtPos  = [posTok[tgtLane+(5*i)] for i in range(len(world.agents))]

        for pos in tgtPos:
            posTok.remove(pos)

        tgtPos  = [np.array(pos) for pos in tgtPos]
        posTok  = [np.array(posTok[i]) for i in np.random.choice(len(posTok),
                                                       len(world.landmarks)+len(world.agents),
                                                       replace=False)]

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if landmark.car:
                landmark.color       = np.array([0.85, 0.35, 0.35])
                landmark.state.p_pos = posTok.pop()
            else:
                landmark.color       = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = tgtPos.pop()
            landmark.state.p_vel     = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.color       = np.array([0.35, 0.85, 0.35])
            agent.state.p_pos = posTok.pop()
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c     = np.zeros(world.dim_c)

        self.render_geoms = None

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

    def is_proximate(self, elem, agent, thresh):
        delta_pos = agent.state.p_pos - elem.state.p_pos
        dist      = np.sqrt(np.sum(np.square(delta_pos)))

        return True if dist < (thresh + agent.size + elem.size) else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in [l for l in world.landmarks if not l.car]:
            dist_rew = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew  -= min(dist_rew)

            goal_rew = [1 if self.is_collision(l, agent) else 0 for l in [l for l in world.landmarks if not l.car]]
            rew  += max(goal_rew)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 10
            for a in [a for a in world.agents if a.name != agent.name]:
                if self.is_proximate(a, agent, 0.02):
                    rew -= 5
            for l in [l for l in world.landmarks if l.car]:
                if self.is_proximate(l, agent, 0.02):
                    rew -= 5

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(200,200)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in world.agents+world.landmarks:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            pos = np.zeros(world.dim_p)
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(world.agents+world.landmarks):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = True))

        return np.concatenate(results)

        # entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)

        # # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
