import numpy as np
from .utils import find_objects, mark_point, repeat_upsample, load_game_dict


n_times = 0

class ColorExtractor():
    """
    Color Extractor module that extracts features.

    """
    def __init__(self, objects_colors=None, game=None):
        """
        Constructor.

        Args:

        """
        if objects_colors is not None:
            self.objects_colors = objects_colors
            self.game = "Unknown"
        elif game is not None:
            try:
                game_dict = load_game_dict(game)
                self.objects_colors = game_dict["colors"].values()
                self.splitted_objects = game_dict["splitted_objects"]
                self.game = game
                self.obj_size = game_dict["obj_size"]
                self.max_obj = game_dict["max_obj"]
                self.img_shape = (210, 160)
                self.divide = max(*self.img_shape)
            except KeyError as err:
                msg = f"\nGame {game} not supported by ColorExtractor\n" + \
                       "please add it to: games_configs.yaml\n"
                print(msg)
                raise err
        else:
            raise ValueError("You need to give objects_colors or game to Color extractor")
        self.show_objects = False

    def __call__(self, images):
        if len(images.shape) == 4:
            batch_size = 1
        elif len(images.shape) == 5:
            batch_size = images.shape[0]
            images = images.reshape(-1, 210, 160, 3)
        all_images_objects = []
        for image in images:
            objects_in_image = []
            for color in self.objects_colors:
                objects_in_image.extend(find_objects(image, color, size=self.obj_size,
                                                     splitted_objects=self.splitted_objects,
                                                     mark_objects=self.show_objects))
            obj_vect = np.array(objects_in_image).flatten()
            all_images_objects.append(np.pad(obj_vect, (0, 2*self.max_obj-len(obj_vect)), mode='constant'))
            # all_images_objects.append(objects_in_image)
        global n_times
        n_times += 1
        if self.show_objects and n_times > 200:
            import matplotlib.pyplot as plt
            _, axes = plt.subplots(2, 2, figsize=(18, 18))
            for image, objs, ax in zip(images, all_images_objects, axes.flatten()):
                ax.imshow(image, interpolation='nearest', aspect="equal")
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()
        if batch_size == 1:
            return np.array(all_images_objects).flatten() / self.divide

        return np.array(all_images_objects).reshape(batch_size, -1) / self.divide
