from PIL import Image, ImageDraw, ImageFilter
import os
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from scipy.spatial import Delaunay
import itertools as it
import noise
import numpy as np
import math
import random


class drawer:
    ratio = 1
    imgx = 1920  # Max width
    imgy = 1080  # Max height
    prob_scale = 100000  # scaling of perlin noise/probabilistic points
    image = None  # Result
    per_image = None  # Perlin image
    per = None  # Perlin 2D array
    constellation_star_array = []  # Array of stars coordinates
    seed = 0  # Seed of the sky

    def __init__(self, seed=0):
        self.imgx = self.imgx * self.ratio
        self.imgy = self.imgy * self.ratio
        self.image = Image.new("RGB", (self.imgx, self.imgy))
        self.seed = seed
        random.seed(self.seed)
        print("SEED:", self.seed)

    def generate_sky(self):
        for x in range(self.imgx):
            for y in range(self.imgy):
                p = random.randint(0, self.prob_scale)
                if p <= int(self.per[x][y] * 0.00001):
                    self.draw_star(x, y, 18)
                    self.constellation_star_array.append((x, y))
                elif p <= int(self.per[x][y] * 0.00004):
                    self.draw_star(x, y, 12)
                    self.constellation_star_array.append((x, y))
                elif p <= int(self.per[x][y] * 0.0002):
                    self.draw_star(x, y, 6)
                elif p <= int(self.per[x][y] * 0.0004):
                    self.draw_star(x, y, 4)

    def generate_constellations(self):
        if not len(self.constellation_star_array):
            return
        # DBSCAN clustering algorithm
        # clustering = DBSCAN(eps=150, min_samples=1).fit(self.constellation_star_array)

        # MeanShift clustering algorithm
        bw = estimate_bandwidth(
            self.constellation_star_array,
            quantile=0.08,
            n_samples=len(self.constellation_star_array),
        )
        print("Estimated Bandwidth:", bw)
        if not bw:
            bw = 200
        clustering = MeanShift(bandwidth=bw).fit(self.constellation_star_array)

        # Finding clusters
        constellation_dict = {}
        for i in range(len(clustering.labels_)):
            label = clustering.labels_[i]
            constellation_dict[label] = constellation_dict.get(label, [])
            constellation_dict[label].append(self.constellation_star_array[i])

        # Removing small found clusters
        for i in [j for j in constellation_dict]:
            if len(constellation_dict[i]) <= 3:
                del constellation_dict[i]

        draw = ImageDraw.Draw(self.image)

        links = {}
        for k, cluster in constellation_dict.items():
            ## Delaunay triangulation
            # http://www.qhull.org/html/qh-optq.htm#qhull
            for triangle in Delaunay(cluster).simplices.copy():
                # Extracting lines from triangles
                # INFO: Values are index of the cluster list, not points coordinates
                links[k] = links.get(k, [])
                for line in [sorted(list(i)) for i in it.combinations(triangle, 2)]:
                    links[k].append(tuple(line))
                # Deleting line duplication
                links[k] = list(dict.fromkeys(links[k]))

            ## Generating links between points
            # Distance calculus between points of each line
            def get_distance(point1, point2):
                return (
                    (point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2
                ) ** (1 / 2)

            ## Graph theory
            links_from_point = {}
            # Generating neigbours from a point
            for pair in links[k]:
                links_from_point[pair[0]] = links_from_point.get(pair[0], [])
                links_from_point[pair[0]].append(pair[1])
                links_from_point[pair[1]] = links_from_point.get(pair[1], [])
                links_from_point[pair[1]].append(pair[0])

            # Spreading Algorithm
            lines = []
            start_point = random.randint(0, len(cluster) - 1)
            used_points = [start_point]
            stack = [start_point]
            while stack:
                cumulativ_dist = 0
                point = stack.pop()
                # Finding the maxmimum local distance
                cumulativ_dist_max = max(
                    (
                        [point, get_distance(cluster[point], cluster[neighbour])]
                        for neighbour in links_from_point[point]
                    ),
                    key=lambda x: x[1],
                )[1]
                while True:
                    # If no neighbour
                    if not [
                        neighbour
                        for neighbour in links_from_point[point]
                        if neighbour not in used_points
                    ]:
                        break
                    # First iteration
                    if cumulativ_dist == 0:
                        stack.append(point)
                    # Finding the minimum local distance
                    target, target_dist = min(
                        (
                            [
                                neighbour,
                                get_distance(cluster[point], cluster[neighbour]),
                            ]
                            for neighbour in links_from_point[point]
                            if neighbour not in used_points
                        ),
                        key=lambda x: x[1],
                    )

                    if (cumulativ_dist + target_dist) <= cumulativ_dist_max:
                        cumulativ_dist += target_dist
                        lines.append((point, target))
                        used_points.append(target)
                        point = target
                    else:
                        break

            # Random added line
            lines.append(links[k][random.randint(0, len(links[k]) - 1)])

            # Plotting
            for line in lines:
                tmp = []
                for label in line:
                    tmp.append(cluster[label])
                draw.line(tmp, fill=(192, 192, 192), width=0)

    def generate_perlin_array(
        self, scale=3000, octaves=6, persistence=0.5, lacunarity=2.0
    ):
        """Generate a Perlin image"""

        shape = (self.imgx, self.imgy)
        arr = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                arr[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=self.seed,
                )

        # Postprocessing
        # Values of the array are between 0 and prob_scale
        max_arr = np.max(arr)
        min_arr = np.min(arr)
        norm_me = lambda x: int(self.prob_scale * (x - min_arr) / (max_arr - min_arr))
        norm_me = np.vectorize(norm_me)
        self.per = norm_me(arr)

        # Plotting
        self.per_image = Image.new("RGB", (self.imgx, self.imgy))
        for y in range(self.imgy):
            for x in range(self.imgx):
                grey = int(self.per[x][y] / self.prob_scale * 255)
                self.per_image.putpixel((x, y), (grey, grey, grey))
                background_color = int(
                    self.per[x][y] / self.prob_scale * 48
                ) + random.randint(-5, 5)
                pos = (self.imgx - x - 1, self.imgy - y - 1)
                self.image.putpixel(pos, (background_color - 20, 0, background_color))
        self.image.filter(ImageFilter.BLUR)

    def draw_star(self, x, y, n):
        def gaussian(x, mu, sig):
            """Gaussian bell function"""
            return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

        def astroid(a, x):
            """Lamé curve"""

            def sqrt3(n):
                return n ** (1 / 3)

            return ((sqrt3(a ** 2) - sqrt3(x ** 2)) ** 3) ** (1 / 2)

        # Generate a random blue color
        colors = [random.randint(192, 255) for i in range(3)]
        # Apply the mathematical functions on a grid of n*n pixels
        for j in range(n):
            for i in range(n):
                intensityX = gaussian(-int(n / 2) + j, 0, n / 6)
                intensityY = gaussian(-int(n / 2) + i, 0, n / 6)
                intensityX *= astroid(int(n / 2), -int(n / 2) + j)
                intensityY *= astroid(int(n / 2), -int(n / 2) + i)

                intensity = (intensityX * intensityY) ** (1 / 2) / int(n / 2)
                grey = int(intensity * 255)
                coord = (
                    (x - int(n / 2) + j) % self.imgx,
                    (y - int(n / 2) + i) % self.imgy,
                )

                pixel = tuple(
                    (max(self.image.getpixel(coord)[i], int(colors[i] * intensity)))
                    for i in range(3)
                )
                # Create the pixel
                self.image.putpixel(coord, tuple(pixel))


if __name__ == "__main__":
    seed = 0
    d = drawer(seed)
    d.generate_perlin_array()
    d.generate_sky()
    d.generate_constellations()
    d.image.show()
    d.image.save("sky_" + str(seed) + ".png", "PNG")
