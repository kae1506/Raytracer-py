import math
import random
from tqdm import tqdm
import numpy as np
from PIL import Image

class Vec3:
    def __init__(self, x ,y ,z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __add__(self, other):
        return Vec3(self.x+other.x,self.y+other.y,self.z+other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vec3(self.x*other,self.y*other,self.z*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __repr__(self):
        return (self.x, self.y, self.z).__repr__()

    def get_mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm(self):
        length = self.get_mag()
        return Vec3(
            self.x / length,
            self.y / length,
            self.z / length
        )
    def clamp(self, low, high):
        return Vec3(
            min(max(self.x, low), high),
            min(max(self.y, low), high),
            min(max(self.z, low), high)
        )

    @classmethod
    def white(self):
        return Vec3(1.0, 1.0, 1.0)

    @classmethod
    def black(self):
        return Vec3(0.0, 0.0, 0.0)

    @classmethod
    def random(self):
        return Vec3(random.random(), random.random(), random.random())

class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir.norm()

    def __repr__(self):
        return f"Origin: {self.origin}, Dir: {self.dir}"

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def get_norm(self, hit_pos):
        return (hit_pos - self.center).norm()

    def intersects(self, ray):
        #print(ray)
        # Does the ray hit the sphere?
        to_sphere = self.center-ray.origin
        #print(to_sphere, ray.dir)
        similarity = to_sphere.dot(ray.dir)
        if similarity < 0:
            return None
        closest_point = ray.origin + ray.dir*similarity
        shortest_line = closest_point - self.center
        y = shortest_line.get_mag()

        if similarity < 0:
            return None

        if y <= self.radius:
            x = math.sqrt(self.radius**2 - y**2)
            return similarity-x
        else:
            return None


class Light:
    def __init__(self, pos, color=Vec3.white()):
        self.pos = pos
        self.color = color


class Material:
    def __init__(self,
                color,
                ambient=0.1,
                diffuse=0.7,
                specular=0.1,
                reflective=0.5):

        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.reflective = reflective

    def color_at(self, scene, shape_hit, hit_pos, hit_normal):
        # print(ray.dir, hit_normal)
        color = self.color * self.ambient
        for light in scene.lights:
            to_light = (light.pos - hit_pos).norm()
            to_cam = (scene.cam - hit_pos).norm()

            # add diffuse model
            color += (
                    self.color
                    * self.diffuse
                    * max(hit_normal.dot(to_light), 0.0)
            )

            # add specular model
            halfway = (to_light + to_cam).norm()
            color += (
                    light.color
                    * self.specular
                    * max((halfway.dot(hit_normal)), 0.0) ** 30
            )

        return color


class Image_Material:
    def __init__(self, root,
                ambient=1.0,
                diffuse=0.5,
                specular=0.5,
                reflective=0.8):
        img = Image.open(root)
        self.img = np.array(img).tolist()

        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.reflective = reflective

    def color_at(self, scene, shape_hit, hit_pos, hit_normal):
        radius = len(self.img)/2

        top = shape_hit.center.x - radius
        left = shape_hit.center.x - radius
        #rint(hit_pos, top, left)
        y = top - hit_pos.y
        x = hit_pos.x - left

        y /= shape_hit.radius*2
        x /= shape_hit.radius * 2

        y *= len(self.img)
        x *= len(self.img)

        col = self.img[int(y)][int(x)]
        color =  Vec3(1-col[0]/255.0, 1-col[1]/255.0, 1-col[2]/255.0)


        color *= self.ambient
        for light in scene.lights:
            to_light = (light.pos - hit_pos).norm()
            to_cam = (scene.cam - hit_pos).norm()

            # add diffuse model
            color += (
                    color
                    * self.diffuse
                    * max(hit_normal.dot(to_light), 0.0)
            )

            # add specular model
            halfway = (to_light + to_cam).norm()
            color += (
                    light.color
                    * self.specular
                    * max((halfway.dot(hit_normal)), 0.0) ** 30
            )

        return color


class Scene:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cam = Vec3(width/2, height/2, -width/2)

        self.shapes = []
        self.lights = []

        self.NUM_S = 30
        self.NUM_L = 20

        for _ in range(self.NUM_S):
            sphere = Sphere(
                center = Vec3(
                    random.random()*width,
                    random.random()*height,
                    width/2+random.random()*width
                ),
                radius = random.random()*width/4,
                material = Material(color=Vec3.random())
            )
            self.shapes.append(sphere)


        for _ in range(self.NUM_L):
            light = Light(pos=Vec3(
                random.random()*width,
                random.random()*height,
                width/2 + random.random()*width
            ))
            self.lights.append(light)


def raytrace(ray, scene, max_bounces, depth=0):
    if depth == 0:
        color = Vec3.black()
    if depth == max_bounces:
        return Vec3.black()

    # find closest shape-ray intersection
    shape_hit = None
    min_dist = math.inf

    for shape in scene.shapes:
        dist = shape.intersects(ray)

        if dist is not None:
            if shape_hit is None:
                shape_hit = shape
                min_dist = dist
            elif dist < min_dist:
                shape_hit = shape
                min_dist = dist
    if shape_hit is None:
        return Vec3.black()
    dist = min_dist

    #print('dist ist ', dist)
    hit_pos = ray.origin + ray.dir*dist
    hit_normal = shape_hit.get_norm(hit_pos)

    # color
    color = shape_hit.material.color_at(scene, shape_hit, hit_pos, hit_normal)

    # bounce ray
    bounce_ray = Ray(
        origin=hit_pos,# + hit_normal * 0.001,
        dir=ray.dir - (2 * ray.dir.dot(hit_normal) * hit_normal)
    )
    #print(ray)
    color += raytrace(bounce_ray, scene, max_bounces, depth + 1)

    return color


def render_scene(scene, max_bounces=3):
    pixels = []
    for y in range(scene.height):
        row = []
        for x in range(scene.width):
            row.append(Vec3.black())
        pixels.append(row)

    for y in tqdm(range(scene.height)):
        for x in range(scene.width):
            target = Vec3(x, y, 0)
            #print(target - scene.cam)
            ray = Ray(origin=scene.cam, dir=target-scene.cam)
            #print(ray)
            pixels[y][x] = raytrace(ray, scene, max_bounces)

    return pixels

def write_as_png(filename, pixels, show_when_done=False):
    width = len(pixels[0])
    height = len(pixels)

    file_image = Image.new('RGB', (width, height), color='black')
    img_pxls = file_image.load()

    for y in range(height):
        for x in range(width):
            col = pixels[y][x]
            col = col.clamp(0,1) * 255.0
            img_pxls[x, height - 1 - y] = (
                int(col.x),
                int(col.y),
                int(col.z)
            )

    file_image.save(filename)
    if show_when_done:
        file_image.show()


def main():
    resolutions = [[20,20], [200,200], [500,500], [1920, 1080], [4096, 2160], [7680, 4320]]

    width, height = resolutions[4]
    scene = Scene(width, height)

    pixels = render_scene(scene)
    write_as_png('render.png', pixels, show_when_done=True)

if __name__ == '__main__':
    main()