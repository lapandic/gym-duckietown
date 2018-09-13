import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.patches as patches
import shapely.geometry as geometry


class RRTStar():
    def __init__(self, start,goal, sample_area, obstacle_list, steer_step = 1, d = 0.01, max_iter = 100):
        self.start = Node(start[0],start[1])
        self.goal = Node(goal[0],goal[1])
        # TODO: find smarter way to sample area
        self.sample_area = sample_area # np.array [xmin xmax ymin ymax]
        self.xmin, self.xmax, self.ymin, self.ymax = sample_area
        self.obstacle_list = obstacle_list

        self.graph = [self.start]

        self.steer_step = steer_step

        self.d = d

        self.max_iter = max_iter

        self.gamma = 50.0 #roughly allows space containing more than 2500 drivable tiles?
        # gamma > (2 (1+1/d))^(1/d) (µ(Xfree)/ζd)^(1/d) > 1.1*sqrt(#drivable tiles)

        self.axis = []


    def planner(self,animation=False):
        if self._is_collision(self.start) or self._is_collision(self.goal):
            return None

        for i in range(self.max_iter):
            # SampleFree
            rand_point = self.get_random_point()
            # Nearest
            nearest_idx = self.get_nearest_node_idx(rand_point)
            nearest_node = self.graph[nearest_idx]
            # Steer
            new_node = self.steer(nearest_idx,rand_point)

            if self.collision_free(nearest_node,new_node):
                card = len(self.graph)*1.0
                r = np.min([self.gamma*np.sqrt(np.log(card)/card),self.steer_step])
                near_nodes_ids = self.get_near_nodes(new_node.point,r)
                self.graph.append(new_node)

                min_node_idx = nearest_idx
                min_cost = nearest_node.cost + self.cost_function(nearest_node,new_node)

                # connect along a minimum-cost path
                for idx in near_nodes_ids:
                    near_node = self.graph[idx]
                    if self.collision_free(near_node,new_node) and near_node.cost + self.cost_function(near_node,new_node) < min_cost:
                        min_node_idx = idx
                        min_cost = near_node.cost + self.cost_function(near_node,new_node)

                self.graph[-1].parent = min_node_idx

                # rewire the tree
                for idx in near_nodes_ids:
                    near_node = self.graph[idx]
                    new_cost = new_node.cost + self.cost_function(new_node,near_node)
                    if self.collision_free(new_node,near_node) and new_cost < near_node.cost:
                        self.graph[idx].parent = len(self.graph)-1
                        self.graph[idx].cost = new_cost

            if animation:
                self.plot(rand_point)

        #TODO: do we need a stopping criteria? if in the goal region?
        last_idx = self.get_last_index()
        if last_idx == None:
            print('Unsuccessful search')
            return None
        else:
            path = self.get_final_path(last_idx)
            if animation:
                plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                plt.grid(True)
                plt.pause(0.01)  # Need for Mac
                plt.show()
            return path


    def plot(self,point):
        """
        Draw Graph
        """
        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(111, aspect='equal')

        for obst in self.obstacle_list:
            ax.add_patch(patches.Polygon(obst.get_polygon(), fill = True))

        plt.plot(self.start.point.x, self.start.point.y, "xr")

        if self.axis == []:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            self.axis = [np.min([xmin,self.xmin]),np.max([xmax,self.xmax]),np.min([ymin,self.ymin]),np.max([ymax,self.ymax])]

        if point is not None:
            plt.plot(point.x, point.y, "^k")

        for node in self.graph:
            if node.parent is not None:
                plt.plot([node.point.x, self.graph[node.parent].point.x], [
                         node.point.y, self.graph[node.parent].point.y], "-g")


        plt.plot(self.goal.point.x, self.goal.point.y, "xr")
        plt.axis(self.axis)

        plt.grid(True)
        plt.pause(0.01)

    def get_final_path(self,last_idx):
        path = [[self.goal.point.x, self.goal.point.y]]
        while self.graph[last_idx].parent is not None:
            node = self.graph[last_idx]
            path.append([node.point.x, node.point.y])
            last_idx = node.parent
        path.append([self.start.point.x, self.start.point.y])
        path.reverse()
        return path

    def get_last_index(self):

        goal_ids = [idx for idx in range(len(self.graph)) if self.euclidian_distance(self.graph[idx].point,self.goal.point) <= self.steer_step]

        if len(goal_ids) == 0:
            return None

        costs = [self.cost_function(self.graph[idx],self.goal) for idx in goal_ids]
        return goal_ids[np.argmin(costs)]

    def cost_function(self,start_node,end_node):
        #TODO: cost function
        #currently Euclidian distance
        return self.euclidian_distance(start_node.point,end_node.point)

    def euclidian_distance(self,start_point,end_point):
        return np.sqrt((end_point.x-start_point.x)**2 +(end_point.y-start_point.y)**2)

    def get_random_point(self):
        xrand = np.random.uniform(self.xmin,self.xmax)
        yrand = np.random.uniform(self.ymin,self.ymax)
        return Point(xrand,yrand)

    def get_nearest_node_idx(self,rand_point):
        costs = [self.cost_function(iter_node,Node(rand_point.x,rand_point.y)) for iter_node in self.graph]
        # bad implementation Node(rand_point.x,rand_point.y)
        return np.argmin(costs)

    def steer(self,nearest_idx,rand_point):
        nearest_node = self.graph[nearest_idx]
        new_node = copy.deepcopy(nearest_node) # has to go with deep copy otherwise it duplicates the existing node in graph
        theta, dist = new_node.get_theta_and_distance(rand_point)
        # TODO: check if this is ok
        # if self.steer_step > dist:
        #     step = self.steer_step/10.0
        # else:
        #     step = self.steer_step
        new_node.point.x += self.steer_step * np.cos(theta)
        new_node.point.y += self.steer_step * np.sin(theta)
        new_node.cost    += self.cost_function(nearest_node,new_node)
        new_node.parent   = nearest_idx
        return new_node

    def get_near_nodes(self,point,r):
        return [idx for idx in range(len(self.graph)) if self.euclidian_distance(self.graph[idx].point,point)**2 < r]

    def is_in_ball(self,point,center,r):
        if self.euclidian_distance(point,center)**2 < r:
            return True
        return False


    def collision_free(self,node1,node2):
        new_node = copy.deepcopy(node1)
         #debug
        plt.plot([new_node.point.x, node2.point.x],[new_node.point.y,node2.point.y], "-r")

        theta, distance = node1.get_theta_and_distance(node2.point)
        for i in range(int(distance/self.d)):
            new_node.point.x += self.d*np.cos(theta)
            new_node.point.y += self.d*np.sin(theta)


            if self._is_collision(new_node):
                return False
        return True

    def _is_collision(self,new_node):
        for obst in self.obstacle_list:
            if obst.check_collision(new_node.point):
                return True
        return False



class Node():
    """
    Class Node
    """
    def __init__(self,x,y):
        self.point = Point(x,y)
        self.cost = 0.0
        self.parent = None

    def get_theta_and_distance(self,point):
        dy = point.y-self.point.y
        dx = point.x-self.point.x
        return np.arctan2(dy,dx),np.sqrt(dx**2 + dy**2)

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def to_tuple(self):
        return (self.x,self.y)

class Obstacle():
    """
    Class Obstacle
    """
    def __init__(self,points,type="undrivable",penalty=10000):
        self.points = points # can be generalized to polygons but we will stick to parallelograms, 4 corners in the clockwise order starting from bottom-left
        self.type = type
        self.penalty = penalty
        self.polygon = geometry.Polygon([self.points[i].to_tuple() for i in range(len(self.points))])

    def check_collision(self,newPoint):
        return self.polygon.contains(geometry.Point(newPoint.to_tuple()))


    def check_collision_old(self,newPoint):
        #works only for parallelograms
        AM_vec = get_vector(self.points[0],newPoint)
        AB_vec = get_vector(self.points[0],self.points[1])
        AD_vec = get_vector(self.points[0],self.points[3])

        if 0 < np.dot(AM_vec,AB_vec) and np.dot(AM_vec,AB_vec) < np.dot(AB_vec,AB_vec):
            if 0 < np.dot(AM_vec,AD_vec) and np.dot(AM_vec,AD_vec) < np.dot(AD_vec,AD_vec):
                return True
        return False

    def expand_bounding_box(self,d):
        self.points[0].x += -d
        self.points[0].y += -d
        self.points[1].x += -d
        self.points[1].y +=  d
        self.points[2].x +=  d
        self.points[2].y +=  d
        self.points[3].x +=  d
        self.points[3].y += -d
        return

    def get_polygon(self):
        x_vec = []
        y_vec = []
        for pt in self.points:
            x_vec.append(pt.x)
            y_vec.append(pt.y)
        return list(zip(x_vec,y_vec))



def get_vector(point1,point2):
    return [point2.x-point1.x,point2.y-point1.y]

def draw_obstacles(obstacle_list,point=Point(0,0),angle=0):
    """
    Draw Graph
    """
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='equal')

    for obst in obstacle_list:
        ax.add_patch(patches.Polygon(obst.get_polygon(), fill = True))

    plt.plot(point.x, point.y, "xr")
    plt.quiver(point.x,point.y,np.cos(angle)*10,np.sin(angle)*10)
    plt.grid(True)
    plt.pause(0.01)





def main():

    show_animation = True

    x = 0.1
    y = 0.1
    obstacle_list = [Obstacle([Point(1,1),Point(1,2),Point(2,2),Point(2,1)]),
                     Obstacle([Point(3.5,3.4),Point(3.3,4.8),Point(5,4.7),Point(4.9,3.6)])]

    #draw_obstacles(obstacle_list,Point(x,y),0)

    rrt_star = RRTStar(start=[x, y], goal=[3, 5],
              sample_area=[-2, 8, -2, 8], obstacle_list=obstacle_list)
    path = rrt_star.planner(animation=show_animation)

    print(path)


if __name__ == "__main__":
    main()



