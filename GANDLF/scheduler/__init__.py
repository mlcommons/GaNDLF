from .triangle import base_triangle, triangle_modified


# defining dict for schedulers - key is the string and the value is the transform object
global_schedulers_dict = {
    "triangle": base_triangle,
    "triangle_modified": triangle_modified,
}
