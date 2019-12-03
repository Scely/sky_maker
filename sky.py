from PIL import Image, ImageDraw
import uuid
import os
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from scipy.spatial import Delaunay
import itertools as it
import noise
import numpy as np
import math
import random

class drawer():
    ratio = 1
    imgx = 1920 # max width
    imgy = 1080 # max height
    prob_scale = 100000 # scaling of perlin noise/probabilistic points
    image = None # Result
    per_image = None # Perlin image
    per = None # Perlin 2D array
    constellation_star_array = [] #Array of star coordinates

    def __init__(self):
        self.imgx = self.imgx * self.ratio
        self.imgy = self.imgy * self.ratio
        self.image = Image.new("RGB", (self.imgx, self.imgy))
        
    def generate_sky(self):
        for x in range(self.imgx):
            for y in range(self.imgy):
                n = 2
                p = int(int.from_bytes(os.urandom(n), "little") * self.prob_scale/2**(n*8))                
                if p < self.per[x][y]*0.00001:
                    self.draw_star(x, y, 18, 255)
                    self.constellation_star_array.append((x, y))
                elif p < self.per[x][y]*0.00006:
                    self.draw_star(x, y, 12, 255)
                    self.constellation_star_array.append((x, y))
                elif p < self.per[x][y]*0.0002:
                    self.draw_star(x, y, 6, 192)
                elif p < self.per[x][y]*0.0004:
                    self.draw_star(x, y, 4, 128)

                    
    def generate_constellations(self):
        if not len(self.constellation_star_array):
            return
        # DBSCAN clustering algorithm
        clustering = DBSCAN(eps=150, min_samples=1).fit(self.constellation_star_array)
        
        # MeanShift clustering algorithm
        #bw = estimate_bandwidth(self.constellation_star_array, quantile=0.15, n_samples=len(self.constellation_star_array))
        #print("Bandwidth estimated:", bw)
        #if not bw:
        #    bw = 400
        #clustering = MeanShift(bandwidth=bw).fit(self.constellation_star_array)
        
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
        
        print("constellation_dict:", constellation_dict)
        links = {}
        for k, cluster in constellation_dict.items():
            ## Delaunay triangulation
            print("k", k)
            print("cluster", cluster)
            for triangle in Delaunay(cluster).simplices.copy():
                # Extracting lines from triangles
                # INFO: Values are index of the cluster list, not points coordinates
                links[k] = links.get(k, [])
                for line in [sorted(list(i)) for i in it.combinations(triangle, 2)]:
                    links[k].append(tuple(line))
                # Deleting line duplication
                links[k] = list(dict.fromkeys(links[k]))
            
            ## Generating links between points
            print("links", links)
            # Distance calculus between points of each line
            max_distance = 0
            tmp = []
            for pair in links[k]:
                p1 = constellation_dict[k][pair[0]]
                p2 = constellation_dict[k][pair[1]]
                distance = ((p1[1]-p2[1])**2 + (p1[0]-p2[0])**2)**(1/2)
                max_distance = max(distance, max_distance)
                tmp.append((pair[0], pair[1], int(distance)))
            links[k] = tmp
                
            # Points number
            n = max(links[k], key= lambda x: x[1])[1]+1 
            # Graph theory
            links_from_point = {}
            # Generating neigbours from a point
            for triple in links[k]:
                links_from_point[triple[0]] = links_from_point.get(triple[0], [])
                links_from_point[triple[0]].append((triple[1], triple[2]))
                links_from_point[triple[1]] = links_from_point.get(triple[1], [])
                links_from_point[triple[1]].append((triple[0], triple[2]))

            # Choice of the first point
            # TODO
            # First point is found with the samllest average distance between all its neigbours
            average_distance = max_distance
            tmp_average_distance = 0
            point = 0
            for k, v in links_from_point.items():
                random.shuffle(v)
                tmp_average_distance = 0
                for pair in v:
                    tmp_average_distance += pair[1]
                tmp_average_distance /= len(v)
                if average_distance > tmp_average_distance:
                    point = k
                    average_distance = tmp_average_distance
            
            # Breadth-first search algorithm
            lines = []
            queue = [point]
            used_points = [point]
            while len(queue):
                s = queue.pop()
                for neighbour in links_from_point[s]:
                    if neighbour[0] not in used_points:
                        used_points.append(neighbour[0])
                        lines.append((s, neighbour[0]))
                        queue.append(neighbour[0])
            print("final lines:", lines)
            
            # Plotting
            for line in lines:
                tmp = []
                for label in line:
                    tmp.append(cluster[label])
                draw.line(tmp, fill=(192, 192, 192), width=0)
                    
        for k, v in links.items():
            print(k, v)


    def generate_perlin_array(
        self,
        scale=3000, octaves = 6, 
        persistence = 0.5, 
        lacunarity = 2.0):
        """Generate a Perlin image"""

        shape = (self.imgx, self.imgy)
        arr = np.zeros(shape)
        seed = int.from_bytes(os.urandom(1), "little")
        for i in range(shape[0]):
            for j in range(shape[1]):
                arr[i][j] = noise.pnoise2(i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    base=seed
                    )
        
        # Postprocessing
        # Values of the array are between 0 and prob_scale
        max_arr = np.max(arr)
        min_arr = np.min(arr)
        norm_me = lambda x: int(self.prob_scale*(x-min_arr)/(max_arr - min_arr))
        norm_me = np.vectorize(norm_me)
        self.per = norm_me(arr)
        
        # Plotting
        self.per_image = Image.new("RGB", (self.imgx, self.imgy))
        for y in range(self.imgy):
            for x in range(self.imgx):
                grey = int(self.per[x][y]/self.prob_scale*255)
                self.per_image.putpixel((x, y), (grey, grey, grey))
                
    
    def draw_star(self, x, y, n, grey):
        def gaussian(x, mu, sig):
            """Gaussian bell function"""
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        def astroid(a, x):
            """Lamé curve"""
            def sqrt3(n):
                return n**(1/3)
            return ((sqrt3(a**2)-sqrt3(x**2))**3)**(1/2)
        # Generate a random blue color
        blue_color = int.from_bytes(os.urandom(1), "little")/3+192
        # Apply the mathematical functions on a grid of n*n pixels
        for j in range(n):
            for i in range(n):
                intensityX = gaussian(-int(n/2)+j, 0, n/6)
                intensityY = gaussian(-int(n/2)+i, 0, n/6)
                intensityX *= astroid(int(n/2), -int(n/2)+j)
                intensityY *= astroid(int(n/2), -int(n/2)+i)
                
                intensity = (intensityX * intensityY)**(1/2)/int(n/2)
                grey = int(intensity*255)
                coord = ((x-int(n/2)+j)%self.imgx, (y-int(n/2)+i)%self.imgy)
                # TODO 
                # Use log to add intensities instead of finding the maximum local value
                pixel = list([max(i, grey) for i in self.image.getpixel(coord)])
                pixel[2] = max(self.image.getpixel(coord)[2], int(blue_color*intensity))
                # Create the pixel
                self.image.putpixel(coord, tuple(pixel))

