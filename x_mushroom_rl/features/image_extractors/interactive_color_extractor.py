import numpy as np
from .utils import find_objects, mark_point, repeat_upsample, load_game_dict
from collections import OrderedDict


class IColorExtractor():
    """
    Interactive Color Extractor module that extracts features.

    """
    def __init__(self, objects_colors=None, game=None, image=None):
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
                # self.max_obj = int(input('What is the maximum number of objects ?'))
                self.max_obj = 20
                self.img_shape = (210, 160)
                self.divide = max(*self.img_shape)
                self.obj_features = 2
                self.imp_objects = ObjDict(self.obj_features, self.max_obj)
                self.n_times = 0
            except KeyError as err:
                msg = f"\nGame {game} not supported by ColorExtractor\n" + \
                       "please add it to: games_configs.yaml\n"
                print(msg)
                raise err
        else:
            raise ValueError("You need to give objects_colors or game to Color extractor")
        self.show_objects = False
        if image is not None:
            self.discarded_zone = discard_zone()


    def __call__(self, images):
        if len(images.shape) == 4:
            batch_size = 1
        elif len(images.shape) == 5:
            batch_size = images.shape[0]
            images = images.reshape(-1, 210, 160, 3)
        all_images_objects = []
        for image in images:
            for color in self.objects_colors:
                objects = find_objects(image, color, size=self.obj_size,
                                       splitted_objects=self.splitted_objects,
                                       mark_objects=self.show_objects)
                self.imp_objects.fill(objects, image)
            objects_vect = self.imp_objects.to_array()
            all_images_objects.append(objects_vect)
        self.n_times += 1
        if self.show_objects and self.n_times > 200:
            import ipdb; ipdb.set_trace()
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


class ObjDict():
    """
    Object (ordered) Dictionary tracking which object has already been seen
    and assigned a class
    """
    def __init__(self, nb_obj_comp, max_obj):
        self.current_obj = OrderedDict()
        self.nb_obj_comp = nb_obj_comp
        self.max_obj = max_obj
        self._vsize = nb_obj_comp*max_obj
        self.object_samples = {}

    def to_array(self):
        if self.current_obj:
            obj_vect = np.array(list(self.current_obj.values())).flatten()
            return np.pad(obj_vect, (0, self._vsize-len(obj_vect)), mode='constant')
        else:
            return np.zeros(self.max_obj * self.nb_obj_comp)

    def empty(self):
        for key in self.current_obj.keys():
            self.current_obj[key] = (0, 0)

    def fill(self, objects_desc, image):
        for position, img_sample in objects_desc:
            already_seen = False
            for classname, sample_list in self.object_samples.items():
                if ObjSample(img_sample) in sample_list:
                    already_seen = True
                    if classname[0] != "_":
                        self.current_obj[classname] = position
                    break
            if not already_seen:
                print("Object has not been seen yet !")
                import matplotlib.pyplot as plt
                f, axarr = plt.subplots(1, 2)
                for ax, img in zip(axarr, [image, img_sample]):
                    ax.imshow(img)
                    ax.xaxis.set_major_locator(plt.NullLocator())
                    ax.yaxis.set_major_locator(plt.NullLocator())
                plt.show()
                inp_msg = "Please provide a class name for this object: \n" + \
                          "HINT: use an underscore (e.g. _score) for unimportant classes: "
                new_classname = input(inp_msg)
                if new_classname in self.object_samples.keys():
                    self.object_samples[new_classname].append(ObjSample(img_sample))
                else:
                    self.object_samples[new_classname] = [ObjSample(img_sample)]
                if new_classname[0] != "_":
                    self.current_obj[new_classname] = position

    def __repr__(self):
        return str(self.object_samples)


class ObjSample():
    def __init__(self, img):
        self.img_sample = img
        self.dominant_color = self._dominant_color()

    def __eq__(self, stored_spl):
        if self.dominant_color != stored_spl.dominant_color:
            return False
        h0, w0, _ = stored_spl.shape
        h1, w1, _ = self.shape
        if w0 != w1 and h0 != h1:
            return False
        elif not self.img_sample.tobytes() == stored_spl.img_sample.tobytes():
            if w0 < w1:  # current sample is smaller and h0 == h1
                if (stored_spl.img_sample == self.img_sample[:, :w0]).all() or \
                   (stored_spl.img_sample == self.img_sample[:, -w0:]).all():
                    print(f"Updating image sample : {stored_spl.shape} -> {self.shape}")
                    show_img([stored_spl.img_sample, self.img_sample])
                    stored_spl.img_sample = self.img_sample
                    stored_spl._dominant_color = self.dominant_color
                    return True
                return False
            elif h0 < h1:  # current sample is smaller and w0 == w1
                if (stored_spl.img_sample == self.img_sample[:h0]).all() or \
                   (stored_spl.img_sample == self.img_sample[-h0:]).all():
                    print(f"Updating image sample : {stored_spl.shape} -> {self.shape}")
                    show_img([stored_spl.img_sample, self.img_sample])
                    stored_spl.img_sample = self.img_sample
                    stored_spl._dominant_color = self.dominant_color
                    return True
                return False
            elif w0 > w1:  # current sample is bigger and h0 == h1
                return (stored_spl.img_sample[:, :w1] == self.img_sample).all() or \
                   (stored_spl.img_sample[:, -w1:] == self.img_sample).all()
            elif h0 > h1:  # current sample is bigger and w0 == w1
                return (stored_spl.img_sample[:h1] == self.img_sample).all() or \
                    (stored_spl.img_sample[-h1:] == self.img_sample).all()

        # only same size image remaining
        for mod_spl in [self.img_sample, np.fliplr(self.img_sample),
                        np.flipud(self.img_sample)]:
            diff = (mod_spl == stored_spl.img_sample)
            ratio = np.count_nonzero(diff) / diff.size
            if ratio >= 0.8:
                return True
        return False

    @property
    def shape(self):
        return self.img_sample.shape

    @property
    def avg_color(self):
        return np.average(self.img_sample, (0,1))

    def _dominant_color(self):
        colors = {}
        for cell in self.img_sample:
            for color in cell:
                hex = '#%02x%02x%02x' % tuple(color)
                if hex in colors:
                    colors[hex] += 1
                else:
                    colors[hex] = 1
        return max(colors, key=colors.get)

    def __repr__(self):
        return f"Object sample: size {self.shape}, dominant color {self.dominant_color}"


def discard_zone(image):
    print("You can discard a zone of the environment, the features extractor would then not look at it.")
    input_msg = "To discard a zone, please input:\n",
    "u (to discard a zone on the UPPER side)\n",
    "b (to discard a zone on the BOTTOM side)\n",
    "r (to discard a zone on the RIGHT side)\n",
    "l (to discard a zone on the LEFT side)\n",
    "Just enter to continue"
    inputed = "None"
    while inputed != "":
        if inputed == "u":
            import ipdb; ipdb.set_trace()
            new_image = image.copy()
        print(msg)
        inputed = input().lower()


def show_img(imgs):
    import matplotlib.pyplot as plt
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    f, axarr = plt.subplots(1, len(imgs))
    for ax, img in zip(axarr, imgs):
        ax.imshow(img)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    plt.show()
