import numpy as np
import time


def point_to_line(pt, line):
    """
    Calculate the shortest distance between a given point and a given line (segment)
    :param pt: a point coordinate where the format is a 2-element iterable list or tuple
    :param line: a "Line" object
    :return: a boolean value saying whether the mathematically shortest connection between
      the point and the line segment can be constructed
      ,shortest distance
    """

    # horizontal case
    if line.gradient == 0:
        inter_x = pt[0]
        inter_y = line.pt1[1]

    # vertical case
    elif line.gradient == np.inf:
        inter_x = line.pt1[0]
        inter_y = pt[1]
        return line.has_point([inter_x, inter_y]), np.abs(pt[0]-line.pt1[0])

    # normal case
    else:
        inter_x = (pt[1] + pt[0]/line.gradient - line.bias) / (line.gradient + 1/line.gradient)
        inter_y = line.gradient * inter_x + line.bias
    distance = np.sqrt((pt[1]-inter_y)**2 + (pt[0]-inter_x)**2)
    return line.has_point([inter_x, inter_y]), distance


def distance_between_points(pt1, pt2):
    """
    Calculate the Euclidean distance between two points as their shortest distance
    :param pt1: a point coordinate where the format is a 2-element iterable list or tuple
    :param pt2: another point coordinate with the same format
    :return: Euclidean distance
    """
    # Euclidean distance
    return np.sqrt((pt1[1] - pt2[1]) ** 2 + (pt1[0] - pt2[0]) ** 2)


class Line(object):
    def __init__(self, pt1, pt2):
        """
        constructor
        :param pt1: a point coordinate
        :param pt2: another point coordinate
        """
        self.pt1, self.pt2 = tuple(pt1), tuple(pt2)

        if pt1[0] < pt2[0]:
            self.left_pt, self.right_pt = pt1, pt2
        else:
            self.left_pt, self.right_pt = pt2, pt1

        if pt1[1] < pt2[1]:
            self.bottom_pt, self.top_pt = pt2, pt1
        else:
            self.bottom_pt, self.top_pt = pt1, pt2

        if pt1[0] == pt2[0]:
            self.gradient = np.inf
        else:
            self.gradient = (pt1[1] - pt2[1])/(pt1[0]-pt2[0])
        self.bias = pt1[1] - self.gradient*pt1[0]
        self.length = np.sqrt((pt1[1] - pt2[1])**2 + (pt1[0]-pt2[0]) ** 2)
        self.nearest_line = None
        self.nearest_distance = np.inf

        if self.gradient is not np.inf:
            self.degree = np.arctan(self.gradient) / np.pi * 180 + 90
            if (self.gradient > 0) and (self.bias < 0):
                self.degree -= 180

            self.distance = np.abs(np.sin(self.degree / 180 * np.pi) * np.abs(self.bias))

        else:
            self.degree = 0
            self.distance = pt1[0]

    def has_point(self, pt):
        """
        Judge whether a given point lies on the line segment
        :param pt: a given point coordinate
        :return: a boolean value illustrating the existence of the point on the line segment
        """
        exist_x = self.left_pt[0] <= pt[0] <= self.right_pt[0]
        exist_y = self.top_pt[1] <= pt[1] <= self.bottom_pt[1]

        if self.gradient == np.inf or self.gradient == -np.inf:
            meet_func = pt[0] == self.pt1[0]
        else:
            # sometimes, there is a loss of significance
            meet_func = pt[0] * self.gradient + self.bias - pt[1] < 0.000000001

        return exist_x and exist_y and meet_func

    def distance_to(self, another):
        """
        Calculate the shortest distance between any line segments
        :param another: another line segment
        :return: shortest distance
        """

        # 2 parallel line segments
        if self.gradient == another.gradient:

            # they are horizontal lines
            if self.gradient == 0:
                if another.left_pt[0] < self.left_pt[0] and another.right_pt[0] < self.left_pt[0]:
                    distance = distance_between_points(another.right_pt, self.left_pt)
                elif another.left_pt[0] > self.right_pt[0] and another.right_pt[0] > self.right_pt[0]:
                    distance = distance_between_points(another.left_pt, self.right_pt)

                else:
                    distance = np.abs(self.right_pt[1] - another.right_pt[1])

            # they are vertical lines
            elif self.gradient is np.inf or self.gradient is -np.inf:
                # the lower pixal infers the larger y
                if another.top_pt[1] > self.bottom_pt[1] and another.top_pt[1] > self.top_pt[1]:
                    distance = distance_between_points(another.top_pt, self.bottom_pt)
                elif another.bottom_pt[1] < self.top_pt[1] and another.bottom_pt[1] < self.bottom_pt[1]:
                    distance = distance_between_points(another.bottom_pt, self.top_pt)

                else:
                    distance = np.abs(self.top_pt[0] - another.bottom_pt[0])

            # 2 parallel line segments with any other gradients
            else:
                b1 = self.right_pt[1] + 1/self.gradient * self.right_pt[0]
                b2 = self.left_pt[1] + 1/self.gradient * self.left_pt[0]

                intersection_x1 = (b1 - another.bias) / (another.gradient + 1/self.gradient)
                intersection_x2 = (b2 - another.bias) / (another.gradient + 1/self.gradient)

                # ======
                if intersection_x1 < another.left_pt[0] and intersection_x2 < another.left_pt[0]:
                    distance = distance_between_points(another.left_pt, self.right_pt)
                elif intersection_x1 > another.right_pt[0] and intersection_x2 > another.right_pt[0]:
                    distance = distance_between_points(another.right_pt, self.left_pt)
                # ————————  ___________
                else:
                    intersection_y = another.gradient * intersection_x1 + another.bias
                    distance = distance_between_points([intersection_x1, intersection_y], self.right_pt)
        else:
            # 2 line segments are not parallel

            # any one is vertical
            if self.gradient is np.inf or self.gradient is -np.inf:
                intersection_x = self.pt1[0]
                intersection_y = another.gradient * intersection_x + another.bias
            # any one is horizontal
            elif another.gradient is np.inf or another.gradient is -np.inf:
                intersection_x = another.pt1[0]
                intersection_y = self.gradient * intersection_x + self.bias
            # normal gradients
            else:
                intersection_x = (another.bias - self.bias) / (self.gradient - another.gradient)
                intersection_y = self.gradient * intersection_x + self.bias

            # the mathematical intersection between two lines, but it might not exist on the line
            intersection = [intersection_x, intersection_y]

            #       |
            #   ————x————
            #       |
            if self.has_point(intersection) and another.has_point(intersection):
                distance = 0

            # |_________
            #           |______
            #
            #  ————————————————————————o——————
            # This case represents the intersection lies on one of the two segments
            elif self.has_point(intersection):
                if intersection[0] < another.left_pt[0]:
                    pt = another.left_pt
                else:
                    pt = another.right_pt
                if self.gradient == 0:
                    inter_x = pt[0]
                    inter_y = self.pt1[1]
                else:
                    inter_x = (pt[1] + pt[0] / self.gradient - self.bias) / (self.gradient + 1 / self.gradient)
                    inter_y = self.gradient * inter_x + self.bias
                if self.has_point([inter_x, inter_y]):
                    distance = distance_between_points(pt, [inter_x, inter_y])
                else:
                    d1 = distance_between_points(pt, self.pt1)
                    d2 = distance_between_points(pt, self.pt2)
                    distance = np.minimum(d1, d2)

            elif another.has_point(intersection):
                if intersection[0] < self.left_pt[0]:
                    pt = self.left_pt
                else:
                    pt = self.right_pt
                if another.gradient == 0:
                    inter_x = pt[0]
                    inter_y = another.pt1[1]
                else:
                    inter_x = (pt[1] + pt[0] / another.gradient - another.bias) / (another.gradient + 1 / another.gradient)
                    inter_y = another.gradient * inter_x + another.bias
                if another.has_point([inter_x, inter_y]):
                    distance = distance_between_points(pt, [inter_x, inter_y])
                else:
                    d1 = distance_between_points(pt, self.pt1)
                    d2 = distance_between_points(pt, self.pt2)
                    distance = np.minimum(d1, d2)

            else:
                # |_________                         ____|
                #           |______               __|
                #
                #                         o
                # this case represents the intersection is out of both line segment. Besides,
                # the intersection locates between two line segments
                if self.right_pt[0] < intersection[0] < another.left_pt[0]:
                    exist1, d1 = point_to_line(self.right_pt, another)
                    exist2, d2 = point_to_line(another.left_pt, self)
                    if exist1 and exist2:
                        distance = np.minimum(d1, d2)
                    elif exist1:
                        distance = d1
                    elif exist2:
                        distance = d2
                    else:
                        distance = distance_between_points(self.right_pt, another.left_pt)

                elif another.right_pt[0] < intersection[0] < self.left_pt[0]:
                    exist1, d1 = point_to_line(self.left_pt, another)
                    exist2, d2 = point_to_line(another.right_pt, self)
                    if exist1 and exist2:
                        distance = np.minimum(d1, d2)
                    elif exist1:
                        distance = d1
                    elif exist2:
                        distance = d2
                    else:
                        distance = distance_between_points(self.left_pt, another.right_pt)

                # |_________
                #           |______
                #
                #  ————————————————————       o
                # this case represents the intersection is out of both line segment. Besides,
                # the line segments locates at the same side of the intersection
                elif self.right_pt[0] < intersection[0] and another.right_pt[0] < intersection[0]:
                    exist1, d1 = point_to_line(self.right_pt, another)
                    exist2, d2 = point_to_line(another.right_pt, self)

                    if exist1 and exist2:
                        distance = np.minimum(d1, d2)
                    elif exist1:
                        distance = d1
                    elif exist2:
                        distance = d2
                    else:
                        distance = distance_between_points(self.right_pt, another.right_pt)

                elif self.left_pt[0] > intersection[0] and another.left_pt[0] > intersection[0]:
                    exist1, d1 = point_to_line(self.left_pt, another)
                    exist2, d2 = point_to_line(another.left_pt, self)

                    if exist1 and exist2:
                        distance = np.minimum(d1, d2)
                    elif exist1:
                        distance = d1
                    elif exist2:
                        distance = d2
                    else:
                        distance = distance_between_points(self.left_pt, another.left_pt)
                elif self.left_pt[0] == intersection[0] == self.right_pt[0]:
                    exist1, d1 = point_to_line(another.right_pt, self)
                    exist2, d2 = point_to_line(another.left_pt, self)

                    if exist1 and exist2:
                        distance = np.minimum(d1, d2)
                    elif exist1:
                        distance = d1
                    elif exist2:
                        distance = d2
                    else:
                        d3 = distance_between_points(self.left_pt, another.left_pt)
                        d4 = distance_between_points(self.right_pt, another.left_pt)
                        d5 = distance_between_points(self.left_pt, another.right_pt)
                        d6 = distance_between_points(self.right_pt, another.right_pt)
                        distance = np.min([d3, d4, d5, d6])
                elif another.left_pt[0] == intersection[0] == another.right_pt[0]:
                    exist1, d1 = point_to_line(self.right_pt, another)
                    exist2, d2 = point_to_line(self.left_pt, another)

                    if exist1 and exist2:
                        distance = np.minimum(d1, d2)
                    elif exist1:
                        distance = d1
                    elif exist2:
                        distance = d2
                    else:
                        d3 = distance_between_points(self.left_pt, another.left_pt)
                        d4 = distance_between_points(self.right_pt, another.left_pt)
                        d5 = distance_between_points(self.left_pt, another.right_pt)
                        d6 = distance_between_points(self.right_pt, another.right_pt)
                        distance = np.min([d3, d4, d5, d6])
                else:
                    distance = -100
        return distance


def test_speed():
    lines = [Line(np.random.randint(0, 1000, size=2),
                  np.random.randint(0, 1000, size=2)) for i in range(100)]

    start = time.time()
    for l1 in lines:
        for l2 in lines:
            if l2 is l1:
                continue
            d = l1.distance_to(l2)
            if l1.nearest_distance > d:
                l1.nearest_distance = d
                l1.nearest_line = l2
            if l2.nearest_distance > d:
                l2.nearest_distance = d
                l2.nearest_line = l1

    end = time.time()
    print('Time cost: {} s'.format(end - start))


if __name__ == '__main__':
    # line1 = Line([0, 0], [1, 2])
    #
    # line2 = Line([2, 2], [2, 1])
    # line3 = Line([2, 1], [3, 0])
    # line4 = Line([2, 3], [2, 1])
    # line5 = Line([3, 2], [4, 2])
    # line6 = Line([2, 2], [1, 3])
    # line7 = Line([3, 3], [4, 2])
    #
    # line8 = Line([0, 0], [0, 2])
    # line9 = Line([1, 0], [1, 2])
    # line10 = Line([1, 2], [1, 10])
    # line11 = Line([1, 3], [1, 10])
    # line12 = Line([3, 3], [4, 3])
    #
    # print(line1.distance_to(line2), "\t 1")
    # print(line1.distance_to(line3), "\t {}".format(np.sin(np.arctan(1/2)+np.pi/4)*np.sqrt(2)))
    # print(line1.distance_to(line4), "\t 1")
    # print(line1.distance_to(line5), "\t 2")
    # print(line1.distance_to(line6), "\t {}".format(1/np.sqrt(2)))
    # print(line1.distance_to(line7), "\t {}".format(np.sqrt(5)))
    #
    # print('\n--------------------------------------\n')
    # print(line8.distance_to(line9), "\t 1")
    # print(line8.distance_to(line10), "\t 1")
    # print(line8.distance_to(line11), "\t {}".format(np.sqrt(2)))
    # print(line8.distance_to(line12), "\t {}".format(np.sqrt(10)))
    test_speed()


